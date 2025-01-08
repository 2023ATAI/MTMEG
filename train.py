import time
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from noise import pnoise3
from torch.cuda import random
from tqdm import trange
from data_gen import load_test_data_for_rnn,load_test_data_for_rnn1,load_train_data_for_rnn,load_train_data_for_rnn1,load_test_data_for_cnn, load_train_data_for_cnn,erath_data_transform,sea_mask_rnn,sea_mask_cnn,sea_mask_rnn1,load_train_data_for_CS,load_test_data_for_CS,load_train_data_for_rnn_nomal,load_test_data_for_rnn_nomal
from loss import NaNMSELoss, NaNMSELoss1
from model import MTLCNN, MMOE, LSTMModel, MTLConvLSTMModel,CrossStitchNet
from REDFLSTM import REDFLSTM,FAMLSTM,EDLSTM
from utils import _plotloss, _plotbox, GetKGE, _boxkge, _boxpcc, GetPCC, GetNSE, _boxnse
from MTAN import MTANmodel
from gauss_model import TransformerVAE

def normalize_parameters(model):
    total = model.aerfa.data + model.gamma.data
    model.aerfa.data /= total
    model.gamma.data /= total

def kl_divergence(mean1, log_var1, mean2, log_var2):
    var1 = torch.exp(log_var1)
    var2 = torch.exp(log_var2)
    kl_div = 0.5 * (log_var2 - log_var1 + (var1 + (mean1 - mean2).pow(2)) / var2 - 1)
    return kl_div.sum()

def train(x,
          y,
          static,
          mask, 
          scaler_x,
          scaler_y,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device,
          num_task=None,
          valid_split=True):
    torch.cuda.manual_seed(SEED)
    random.seed()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    patience = cfg['patience']
    wait = 0
    best = 9999
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static[0].dtype))
    if cfg['modelname'] in ['CNN', 'ConvLSTM','MTLCNN','MTLConvLSTMModel','MTANmodel','CrossStitchNet']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    if valid_split:
        nt,nlat,nlon,nf = x.shape  #x shape :nt,nf,nlat,nlon
	#Partition validation set and training set
        N = int(nt*cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], list(static)
        x, y = x[:N], y[:N]       

    lossmse = torch.nn.MSELoss()
#	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static[0].shape)
    print('mask shape is', mask.shape)

    # mask see regions
    #Determine the land boundary
    if cfg['modelname'] in ['MHLSTMModel','MSLSTMModel',"SoftMTLv1","MMOE","LSTM","REDFLSTM","FAMLSTM","EDLSTM"]:
        if valid_split:
            x_valid, y_valid ,static_valid,mask_index_val = sea_mask_rnn1(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static,mask_index = sea_mask_rnn1(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['CNN','ConvLSTM','MTLCNN','MTLConvLSTMModel','CrossStitchNet']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['MTANmodel']:
        if valid_split:
            x_valid_lstm, y_valid_lstm ,static_valid_lstm,mask_index_val = sea_mask_rnn1(cfg, x_valid, y_valid, static_valid, mask)
        x_lst, y_lst, static_lst,mask_index = sea_mask_rnn1(cfg, x, y, static, mask)
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)
    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for num_ in range(1):
        # prepare models
	#Selection model
        # if cfg['modelname'] in [ 'MSLSTMModel']:
        #     mtllstmmodel_cfg = {}
        #     mtllstmmodel_cfg['input_size'] = cfg["input_size"]
        #     mtllstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
        #     mtllstmmodel_cfg['out_size'] = 1
        #     model = MSLSTMModel(cfg,mtllstmmodel_cfg).to(device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if cfg['modelname'] in ['MMOE']:
            MMOEl_cfg = {}
            MMOEl_cfg['input_size'] = cfg["input_size"]
            MMOEl_cfg['hidden_size'] = cfg["hidden_size"]*1
            MMOEl_cfg['out_size'] = 1
            model = MMOE(cfg,MMOEl_cfg).to(device)
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {}
            lstmmodel_cfg['input_size'] = cfg["input_size"]
            lstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
            lstmmodel_cfg['out_size'] = 1
        if cfg['modelname'] in ['REDFLSTM']:
            REDFLSTM_cfg = {}
            REDFLSTM_cfg['input_size'] = cfg["input_size"]
            REDFLSTM_cfg['hidden_size'] = cfg["hidden_size"]*1
            REDFLSTM_cfg['out_size'] = 1
            model = REDFLSTM(cfg,REDFLSTM_cfg).to(device)
        if cfg['modelname'] in ['FAMLSTM']:
            REDFLSTM_cfg = {}
            REDFLSTM_cfg['input_size'] = cfg["input_size"]
            REDFLSTM_cfg['hidden_size'] = cfg["hidden_size"]*1
            REDFLSTM_cfg['out_size'] = 1
            model = FAMLSTM(cfg,REDFLSTM_cfg).to(device)
        if cfg['modelname'] in ['EDLSTM']:
            REDFLSTM_cfg = {}
            REDFLSTM_cfg['input_size'] = cfg["input_size"]
            REDFLSTM_cfg['hidden_size'] = cfg["hidden_size"]*1
            REDFLSTM_cfg['out_size'] = 1
            model = EDLSTM(cfg,REDFLSTM_cfg).to(device)
        elif cfg['modelname'] in ['MTLCNN']:
            model = MTLCNN(cfg).to(device)
        elif cfg['modelname'] in ['MTLConvLSTMModel']:
            model = MTLConvLSTMModel(cfg).to(device)
        elif cfg['modelname'] in ['MTANmodel']:
            model = MTANmodel(cfg).to(device)
            gauss_modelQ = TransformerVAE(feature_dim=128,hidden_dim=cfg["hidden_size"],num_heads=2,num_layers=2,dropout_rate=cfg["dropout_rate"]).to(device)
            gauss_modelP = TransformerVAE(feature_dim=128,hidden_dim=cfg["hidden_size"],num_heads=2,num_layers=2,dropout_rate=cfg["dropout_rate"]).to(device)
        elif cfg['modelname'] in ['CrossStitchNet']:
            model = CrossStitchNet(cfg).to(device)

        # if cfg['modelname'] in [ 'MHLSTMModel']:
        #     mtllstmmodel_cfg = {}
        #     mtllstmmodel_cfg['input_size'] = cfg["input_size"]
        #     mtllstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
        #     mtllstmmodel_cfg['out_size'] = 1
        #     model = MHLSTMModel(cfg, mtllstmmodel_cfg).to(device)

      #  model.train()
	 # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively
    #       learing rate schedule. We found `Adam` perform
    #       much better than `Adagrad`, `Adadelta`.
        if cfg["modelname"] in \
                    ['SoftMTLv1']:
            optimS = torch.optim.Adam([
                {'params': model1.parameters()},
                {'params': model2.parameters()}
            ], lr=cfg['learning_rate'])
        # optimH = torch.optim.Adam([
        #     {'params': model.lstm1.parameters()},
        #     {'params': model.drop.parameters()}
        # ], lr=cfg['learning_rate'])
    #     optimizer = torch.optim.Adam([{'params': model1.lstm1.parameters()}], lr=0.001)
    #     optim1 = torch.optim.Adam(model1.parameters(),lr=cfg['learning_rate'])
    #     optim2 = torch.optim.Adam(model2.parameters(),lr=cfg['learning_rate'])

        if cfg["modelname"] in \
                ['MMOE']:
            # optim1M = torch.optim.Adam(model.tower_layers[0].parameters(),lr=cfg['learning_rate'])
            # optim2M = torch.optim.Adam(model.tower_layers[1].parameters(),lr=cfg['learning_rate'])
            optim1M = torch.optim.Adam([
                {'params': model.tower_layers[0].parameters()},
                {'params': model.gates[0].parameters()}
            ], lr=cfg['learning_rate'])
            optim2M = torch.optim.Adam([
                {'params': model.tower_layers[1].parameters()},
                {'params': model.gates[1].parameters()}
            ], lr=cfg['learning_rate'])
            parameters_to_optimize = []
            for expert_model in model.Expers:
                parameters_to_optimize += list(expert_model.parameters())
            optimSM = torch.optim.Adam(parameters_to_optimize, lr=cfg['learning_rate'])

        # if cfg["modelname"] in \
        #             ['MTLCNN']:
        #     optimH = torch.optim.Adam([
        #         {'params': model.cnn.parameters()},
        #         {'params': model.drop.parameters()}
        #     ], lr=cfg['learning_rate'])
        #     optimizers = []
        #     for i in range(cfg['num_repeat']):  
        #         optimizer = torch.optim.Adam(model.head_layers[i].parameters(), lr=cfg['learning_rate'])  
        #         optimizers.append(optimizer)
        # if cfg["modelname"] in \
        #             ['MTLConvLSTMModel']:
        #     optimH = torch.optim.Adam([
        #         {'params': model.ConvLSTM_net.parameters()},
        #         {'params': model.drop.parameters()}
        #     ], lr=cfg['learning_rate'])
        #     optimizers = []
        #     for i in range(cfg['num_repeat']):  
        #         optimizer = torch.optim.Adam(model.head_layers[i].parameters(), lr=cfg['learning_rate']) 
        #         optimizers.append(optimizer)
        if cfg["modelname"] in \
                    ["LSTM","MTLCNN","MTLConvLSTMModel","CrossStitchNet","MMOE","REDFLSTM","FAMLSTM","EDLSTM"]:
            optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        if cfg["modelname"] in \
                [ "MTANmodel"]:
            sigma = []

            log_sigma1 = torch.zeros(1, requires_grad=True,device = device)
            log_sigma2 = torch.zeros(1, requires_grad=True,device = device)
            optim = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': log_sigma1},
                {'params': log_sigma2}
            ])
            optim_gaussQ = torch.optim.Adam(gauss_modelQ.parameters(),lr=cfg['learning_rate'])
            optim_gaussP = torch.optim.Adam(gauss_modelP.parameters(),lr=cfg['learning_rate'])

        # if cfg["modelname"] in \
        #         ["MTANmodel"]:
        #     optimizer_shared = torch.optim.Adam([
        #         {'params':model.module1.parameters()},
        #         {'params':model.module2.parameters()},
        #         {'params':model.convtran.parameters()},
        #         {'params':model.share_aten.parameters()},
        #         {'params':model.lstm1.parameters()},
        #     ],lr=cfg['learning_rate'])
        #     optimizer_task1 = torch.optim.Adam([
        #         {'params': model.lstm_layers1[0].parameters()},
        #         # {'params': model.att_layers[0].parameters()},
        #         {'params': model.head_layers[0].parameters()}
        #     ],lr=cfg['learning_rate'])
        #     optimizer_task2 = torch.optim.Adam([
        #         {'params': model.lstm_layers1[1].parameters()},
        #         # {'params': model.att_layers[1].parameters()},
        #         {'params': model.head_layers[1].parameters()}
        #     ],lr=cfg['learning_rate'])
        #     optimizer_task = []
        #     optimizer_task.append(optimizer_task1)
        #     optimizer_task.append(optimizer_task2)

        epoch_losses1 = []
        epoch_losses2 = []
        Gauss = True
        with trange(1, cfg['epochs']+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname']+' '+str(num_repeat))
                t_begin = time.time()
                # train
                MSELoss = 0
                for iter in range(0, cfg["niter"]):

 # ------------------------------------------------------------------------------------------------------------------------------
 #  train way for LSTM model


                    if cfg["modelname"] in \
                            ['MSLSTMModel']:

                        x_batch,y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn1(cfg, x, y, static, scaler_y)

                        yy = []
                        xx = []
                        aux = []
                        for i in range(len(y_batch)):
                            pred_time = torch.from_numpy(y_batch[i]).to(device)
                            x_time = torch.from_numpy(x_batch[i]).to(device)
                            aux_time = torch.from_numpy(aux_batch[i]).to(device)
                            aux_time = aux_time.unsqueeze(1)
                            yy.append(pred_time)

                            aux_time = aux_time.repeat(1, x_time.shape[1], 1)
                            # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                            # print('x_batch[:,5,0]',x_batch[:,5,0])
                            # x_batch =torch.Tensor(w)
                            x_time =torch.cat([x_time, aux_time], 2)
                            xx.append(x_time)
                            aux.append(aux_time)
                        pred = model(xx, aux)
                        #  64的shape
                        pp = []
                        for i in pred:
                            i = i.squeeze()
                            pp.append(i)

                    if cfg["modelname"] in \
                            ['FAMLSTM',"EDLSTM"]:  #FAMLSTM,REDFLSTM
                        x_batch,y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn_nomal(cfg, x, y, static, scaler_y , mask_index)

                        y_lstm = []
                        xx = []
                        aux = []
                        for i in range(len(y_batch)):
                            pred_time = torch.from_numpy(y_batch[i]).to(device)
                            x_time = torch.from_numpy(x_batch[i]).to(device)
                            aux_time = torch.from_numpy(aux_batch[i]).to(device)
                            aux_time = aux_time.unsqueeze(1)
                            y_lstm.append(pred_time)

                            aux_time = aux_time.repeat(1, x_time.shape[1], 1)
                            # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                            # print('x_batch[:,5,0]',x_batch[:,5,0])
                            # x_batch =torch.Tensor(w)
                            x_time =torch.cat([x_time, aux_time], 2)
                            xx.append(x_time)
                            aux.append(aux_time)
                        pred = model(xx, aux,cfg)


                    # MMoe  
                    if cfg["modelname"] in \
                            ['MMOE']:
                        x_batch,y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn1(cfg, x, y, static, scaler_y)

                        xx = []
                        for i in range(len(y_batch)) :
                            y_batch[i] = torch.from_numpy(y_batch[i]).to(device)
                            w = torch.from_numpy(x_batch[i]).to(device)
                            a = torch.from_numpy(aux_batch[i]).to(device)
                            a = a.unsqueeze(1)

                            a = a.repeat(1, w.shape[1], 1)
                            # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                            # print('x_batch[:,5,0]',x_batch[:,5,0])
                            # x_batch =torch.Tensor(w)
                            w =torch.cat([w, a], 2)
                            xx.append(w)
                        pred = model(xx, aux_batch)
                        #  64的shape



                    if cfg["modelname"] in \
                            ['LSTM']:
                        # generate batch data for Recurrent Neural Network
                        if isinstance(y, list):
                            y = y[0]
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_rnn(cfg, x, y, static, scaler_y)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                        #print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        #print('x_batch[:,5,0]',x_batch[:,5,0])
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred = model(x_batch, aux_batch)
                        # print('pred1',pred.shape)
                        pred = torch.squeeze(pred,1)

                    elif cfg['modelname'] in ['MTLCNN']:
                        # generate batch data for Convolutional Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index)
                        predlist = []
                        for pre_num in range(cfg['num_repeat']):
                            x_batch[pre_num][np.isnan(x_batch[pre_num])] = 0  # filter nan values to train cnn model
                            x_batch[pre_num] = torch.from_numpy(x_batch[pre_num]).to(device)
                            aux_batch[pre_num] = torch.from_numpy(aux_batch[pre_num]).to(device)
                            y_batch[pre_num] = torch.from_numpy(y_batch[pre_num]).to(device)
                            x_batch[pre_num] = x_batch[pre_num].squeeze(dim=1)
                            x_batch[pre_num] = x_batch[pre_num].reshape(x_batch[pre_num].shape[0],x_batch[pre_num].shape[1]*x_batch[pre_num].shape[2],x_batch[pre_num].shape[3],x_batch[pre_num].shape[4])
                            x_batch[pre_num] = torch.cat([x_batch[pre_num], aux_batch[pre_num]], 1)
                        pred = model(x_batch, aux_batch)
                    elif cfg['modelname'] in ['CrossStitchNet']:
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_CS(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)
                        predlist = []
                        for pre_num in range(cfg['num_repeat']):
                            x_batch[pre_num][np.isnan(x_batch[pre_num])] = 0  # filter nan values to train cnn model
                            x_batch[pre_num] = torch.from_numpy(x_batch[pre_num]).to(device)
                            aux_batch[pre_num] = torch.from_numpy(aux_batch[pre_num]).to(device)
                            y_batch[pre_num] = torch.from_numpy(y_batch[pre_num]).to(device)
                            x_batch[pre_num] = x_batch[pre_num].squeeze(dim=1)
                            x_batch[pre_num] = x_batch[pre_num].reshape(x_batch[pre_num].shape[0],
                                                                        x_batch[pre_num].shape[1] * x_batch[pre_num].shape[2],
                                                                        x_batch[pre_num].shape[3], x_batch[pre_num].shape[4])
                            x_batch[pre_num] = torch.cat([x_batch[pre_num], aux_batch[pre_num]], 1)
                        pred = model(x_batch, aux_batch)
                    elif cfg['modelname'] in ['MTLConvLSTMModel']:
                        # generate batch data for Convolutional Neural Network
                        x_batch, y_batch, aux_batch, _, _ = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,lat_index,lon_index,mask_index)
                        predlist = []
                        for pre_num in range(cfg['num_repeat']):
                            x_batch[pre_num][np.isnan(x_batch[pre_num])] = 0  # filter nan values to train cnn model
                            x_batch[pre_num] = torch.from_numpy(x_batch[pre_num]).to(device)
                            aux_batch[pre_num] = torch.from_numpy(aux_batch[pre_num]).to(device)
                            y_batch[pre_num] = torch.from_numpy(y_batch[pre_num]).to(device)
                            aux_batch[pre_num] = aux_batch[pre_num].unsqueeze(1)
                            aux_batch[pre_num] = aux_batch[pre_num].repeat(1, x_batch[pre_num].shape[1], 1, 1, 1)
                            x_batch[pre_num] = x_batch[pre_num].squeeze(dim=1)
                        pred = model(x_batch, aux_batch,cfg)

                    elif cfg['modelname'] in ['MTANmodel']:
                        # generate batch data for Convolutional Neural Network
                        x_cnn, y_cnn, aux_cnn, x_lstm, y_lstm,aux_lstm,x_ture = \
                            load_train_data_for_cnn(cfg, x, y, static, scaler_y,x_lst, y_lst, static_lst,lat_index,lon_index,mask_index)

                        predictions = []

                        for pre_num in range(cfg['num_repeat']):
                            # 
                            if Gauss:

                                x_ture[pre_num] = torch.from_numpy(x_ture[pre_num]).to(device)

                                x_lstm[pre_num] = torch.from_numpy(x_lstm[pre_num]).to(device)
                                y_lstm[pre_num] = torch.from_numpy(y_lstm[pre_num]).to(device)
                                aux_lstm[pre_num] = torch.from_numpy(aux_lstm[pre_num]).to(device)
                                aux_lstm[pre_num] = aux_lstm[pre_num].unsqueeze(1)

                                aux_ture = aux_lstm[pre_num].repeat(1, x_ture[pre_num].shape[1], 1)

                                aux_lstm[pre_num] = aux_lstm[pre_num].repeat(1, x_lstm[pre_num].shape[1], 1)
                                x_lstm[pre_num] = torch.cat([x_lstm[pre_num], aux_lstm[pre_num]], 2)

                                x_ture[pre_num] = torch.cat([x_ture[pre_num], aux_ture], 2)

                                inputs = x_lstm[pre_num]
                                true = x_ture[pre_num]

                                optim_gaussP.zero_grad()
                                mean1, log_var1 = gauss_modelP(inputs)
                                with torch.no_grad():
                                    Flag = False
                                    mean2, log_var2 = gauss_modelQ(true,Flag)

                                kl_loss = kl_divergence(mean1, log_var1, mean2[:,1:,:], log_var2[:,1:,:])
                                kl_loss.backward()
                                optim_gaussP.step()

                                std = torch.exp(0.5 * log_var1)

                                epsilon = torch.randn_like(std)
                                # 
                                sampled_noise = mean1 + epsilon * std
                        pred = model(sampled_noise,x_lstm, aux_cnn, cfg)



 #  train way for CNN model
 # ------------------------------------------------------------------------------------------------------------------------------


                    if cfg["modelname"] in ['MMOE']:

                        losses = []
                        loss_sum = 0.0
                        for i in range(cfg['num_repeat']):
                            loss_time = NaNMSELoss.fit(cfg, pred[i].squeeze(dim=1).float(), y_batch[i].float(), lossmse)
                            losses.append(loss_time)
                            loss_sum += loss_time
                        loss = loss_sum/cfg['num_repeat']

                        optimSM.zero_grad()
                        optim1M.zero_grad()
                        optim2M.zero_grad()

                        loss.backward(retain_graph=True)
                        losses[0].backward(retain_graph=True)
                        losses[1].backward(retain_graph=True)

                        optimSM.step()
                        optim1M.zero_grad()
                        optim2M.zero_grad()
                    #


                    # elif cfg["modelname"] in ['MSLSTMModel']:
                    #     losses = []
                    #
                    #     for i in range(cfg['num_repeat']):
                    #         loss_time = NaNMSELoss.fit(cfg, pp[i].float(), yy[i].float(), lossmse)
                    #         losses.append(loss_time)
                    #
                    #     optimH.zero_grad()
                    #     # 权重分配！
                    #     loss = 0.2 * losses[0] + 0.8 * losses[1]
                    #     # weight = F.softmax(torch.randn(cfg['num_repeat']), dim=-1)
                    #     # weight = weight.to(device)
                    #     # Random Loss Weighting  簡稱RLW
                    #     # loss 包含所有的損失 與隨機分配的權重張量相乘
                    #     # loss = sum(losses[i] * weight[i] for i in range(cfg['num_repeat']))
                    #     loss.backward(retain_graph=True)
                    #     for i in range(cfg['num_repeat']):
                    #         op = optimizers[i]
                    #         op.zero_grad()
                    #         losses[i].backward(retain_graph=True)
                    #         op.step()
                    #     optimH.step()

                    elif cfg["modelname"] in ['LSTM']:
                        loss = NaNMSELoss1.fit(cfg, pred.float(), y_batch.float(), lossmse)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        MSELoss += loss.item()

                    # elif cfg['modelname'] in ['MTLCNN','MTLConvLSTMModel']:
                    #     losses = []
                    #     loss_sum = 0.0
                    #
                    #     for i in range(cfg['num_repeat']):
                    #         loss_time = NaNMSELoss.fit(cfg, pred[i].squeeze(dim=1).float(), y_batch[i].float(), lossmse)
                    #         losses.append(loss_time)
                    #         loss_sum += loss_time
                    #     loss = loss_sum/cfg['num_repeat']
                    #     optimH.zero_grad()
                    #     loss.backward(retain_graph=True)
                    #     for i in range(cfg['num_repeat']):
                    #         op = optimizers[i]
                    #         op.zero_grad()
                    #         losses[i].backward(retain_graph=True)
                    #         op.step()
                    #     optimH.step()

                    elif cfg['modelname'] in ['MTLCNN','MTLConvLSTMModel',"CrossStitchNet","MTANmodel","REDFLSTM","FAMLSTM","EDLSTM"]:
                        losses = []
                        loss_sum = 0.0
                        optim.zero_grad()
                        for i in range(cfg['num_repeat']):
                            pred_ = pred[i].squeeze().float()
                            loss_time = NaNMSELoss.fit(cfg,pred_, y_lstm[i].float(), lossmse)
                            losses.append(loss_time)
                            loss_sum = loss_sum + loss_time
                        # UW
                        # weighted_loss_task1 = torch.exp(-log_sigma1) * losses[0] + log_sigma1
                        # weighted_loss_task2 = torch.exp(-log_sigma2) * losses[1] + log_sigma2
                        # loss_sum = weighted_loss_task1 + weighted_loss_task2
                        loss = loss_sum/cfg['num_repeat']

                        # normalize_parameters(model)
                        loss.backward()
                        optim.step()
                        MSELoss += loss.item()
                    #     梯度手术
                    # elif cfg['modelname'] in ["MTANmodel"]:
                    #     threshold = 0.1
                    #     # 清空之前的梯度
                    #     losses = []
                    #     optim.zero_grad()
                    #
                    #     for i in range(cfg['num_repeat']):
                    #         pred_ = pred[i].squeeze().float()
                    #         loss_time = NaNMSELoss.fit(cfg, pred_, y_lstm[i].float(), lossmse)
                    #         losses.append(loss_time)
                    #
                    #     # 单独计算两个任务的梯度
                    #     loss1 = losses[0]
                    #     loss2 = losses[1]
                    #
                    #     loss1.backward(retain_graph=True)
                    #     grads_task1 = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for
                    #                    param in model.parameters()]
                    #     optim.zero_grad()
                    #
                    #     loss2.backward(retain_graph=True)
                    #     grads_task2 = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for
                    #                    param in model.parameters()]
                    #     optim.zero_grad()
                    #
                    #     # 梯度手术开始，对每一组梯度进行处理
                    #     for g1, g2 in zip(grads_task1, grads_task2):
                    #         dot_product = torch.dot(g1.view(-1), g2.view(-1))
                    #         # 如果两个梯度的方向相反（即点积为负），并且相反的程度超过了阈值，则对梯度进行修剪
                    #         if dot_product < -threshold:
                    #             # 计算修剪因子
                    #             alpha = dot_product / (g1.norm() ** 2 + g2.norm() ** 2 + 1e-10)
                    #             # 更新梯度
                    #             g1.add_(alpha * g2)
                    #             g2.add_(alpha * g1)
                    #
                    #     # 计算总损失
                    #     total_loss = loss1 + loss2
                    #
                    #     # 清空之前的梯度
                    #     optim.zero_grad()
                    #     # 反向传播，计算对修剪后的梯度进行更新
                    #     total_loss.backward()
                    #     # 更新模型参数
                    #     optim.step()


                    #     多天的迭代预报
                    # elif cfg['modelname'] in ["MTANmodel"]:
                    #     losses1 = []
                    #     losses2 = []
                    #     loss_sum = 0.0
                    #     optim.zero_grad()
                    #     for step in range(cfg["multi_step"]):
                    #         for i in range(cfg['num_repeat']):
                    #             pred_ = predictions[step][i].squeeze().float()
                    #             loss_time = NaNMSELoss.fit(cfg,pred_, y_lstm[i][:,step].float(), lossmse)
                    #             if i == 0:
                    #                 losses1.append(loss_time)
                    #             else:
                    #                 losses2.append(loss_time)
                    #             loss_sum = loss_sum + loss_time
                    #     # UW
                    #     # weighted_loss_task1 = torch.exp(-log_sigma1) * losses[0] + log_sigma1
                    #     # weighted_loss_task2 = torch.exp(-log_sigma2) * losses[1] + log_sigma2
                    #     # loss_sum = weighted_loss_task1 + weighted_loss_task2
                    #     loss = loss_sum/cfg['num_repeat']
                    #
                    #     # normalize_parameters(model)
                    #     loss.backward()
                    #     optim.step()
                    #     MSELoss += loss.item()

                    # mtan在做独立参数更新时的操作
                    # gradnorm
                    # elif cfg['modelname'] in ['MTANmodel']:
                    #
                    #     loss_time = []
                    #     for i in range(cfg['num_repeat']):
                    #         pred_ =  pred[i].squeeze().float()
                    #         loss_time.append(NaNMSELoss.fit(cfg, pred_, y_lstm[i].float(), lossmse))
                    #
                    #     task_loss = torch.stack(loss_time)
                    #     if epoch == 1:
                    #         initial_task_loss = task_loss.data
                    #     weighted_task_loss = torch.mul(model.weights,task_loss)
                    #
                    #     # losses.append(weighted_task_loss)
                    #
                    #     loss_total = torch.sum(weighted_task_loss)
                    #     # clear the gradients
                    #     optim.zero_grad()
                    #     # do the backward pass to compute the gradients for the whole set of weights
                    #     # This is equivalent to compute each \nabla_W L_i(t)
                    #     loss_total.backward(retain_graph=True)
                    #     # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
                    #     model.weights.grad.data = model.weights.grad.data * 0.0
                    #
                    #     if cfg["Grad_norm"]:
                    #         W = model.get_last_shared_layer()
                    #
                    #         norms = []
                    #         for i in range(len(task_loss)):
                    #             # get the gradient of this task loss with respect to the shared parameters
                    #             gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    #             # compute the norm
                    #             norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
                    #         norms = torch.stack(norms)
                    #         loss_ratio = task_loss / initial_task_loss
                    #         inverse_train_rate = loss_ratio / torch.mean(loss_ratio)
                    #         # 计算平均梯度范数 mean_norm
                    #         mean_norm = torch.mean(norms)
                    #         # 计算 GradNorm loss  0.12这个值 若变大则更加依赖相较于初始损失的比率
                    #         constant_term = mean_norm * (inverse_train_rate ** 0.12)
                    #         grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                    #
                    #         # 计算共享参数的梯度
                    #         model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
                    #         # torch.clamp_min(model.weights,0.0)
                    #
                    #     optim.step()
                    #     # 确保每个任务的权重对总体的影响比例相等
                    #     normalize_coeff = 1 / torch.sum(model.weights.data, dim=0)
                    #     model.weights.data = model.weights.data * normalize_coeff

                # ------------------------------------------------------------------------------------------------------------------------------
                if cfg["modelname"] in ['MSLSTMModel','MHLSTMModel','SoftMTLv1','MMOE']:
                    t_end = time.time()
                    # get loss log
                    loss_str = "Epoch {} Loss {:.3f};Loss1 {:.3f};time {:.2f}".format(
                        epoch, loss,losses[0],
                        t_end - t_begin)
                    # loss_str1 = "Epoch {} Train loss1 Loss {:.3f} time {:.2f}".format(epoch, loss1 / cfg["niter"],
                    #                                                                t_end - t_begin)
                    # loss_str2 = "Epoch {} Train loss2 Loss {:.3f} time {:.2f}".format(epoch, loss2 / cfg["niter"],
                    #                                                                t_end - t_begin)
                    print(loss_str)
                elif cfg["modelname"] in ['MTLCNN','MTLConvLSTMModel','MTANmodel','CrossStitchNet',"REDFLSTM","FAMLSTM","EDLSTM"]:
                    # 打印每个任务的损失值和训练时间
                    t_end = time.time()
                    # if losses[0].data > 0.3:
                    #     print(epoch)
                    print(f'Epoch {epoch + 1}, Losses: ', end='')
                    # print("weight",model.weights[0].data,model.weights[1].data)
                    for i, loss_time in enumerate(losses):
                        print(f'Task {i + 1}: {loss_time.item():.4f}', end=' | ')

                    # 定义你想打印的天数索引，注意Python索引是从0开始的，所以减去1
                    # days_to_print_indices = [0, 2, 6]  # 对应于第1天，第3天和第7天
                    #
                    # for index in days_to_print_indices:
                    #     if index < len(losses1) and index < len(losses2):
                    #         # 直接使用index访问对应天数的损失
                    #         loss1 = losses1[index].item()
                    #         loss2 = losses2[index].item()
                    #         # 打印对应天数的损失，天数为索引加1
                    #         print(f'Day {index + 1}: Loss1: {loss1:.4f} - Loss2: {loss2:.4f}')
                    print(f'Time: {(t_begin - t_end):.2f} seconds')
                else:
                    t_end = time.time()
                    # get loss log
                    loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"],
                                                                                   t_end - t_begin)
                    print(loss_str)

                # validate
		#Use validation sets to test trained models
		#If the error is smaller than the minimum error, then save the model.
                if valid_split:
                    # del x_lstm, y_lstm, aux_lstm,x_cnn,y_cnn,aux_cnn
                    MSE_valid_loss = 0
                    if epoch % 10 == 0:

                        wait += 1
                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------
 #  validate way for LSTM model
                        if cfg["modelname"] in ['MSLSTMModel']:
                            gt_list = [i for i in range(0,x_valid[0].shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid[0].shape[0]-cfg["seq_len"])//cfg["stride"]

                            losses_val = []

                            for i in range(0, n):
                                #mask
                                x_valid_batch,y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn1(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                yy = []
                                xx = []
                                pred_valid = []
                                for ii in range(len(y_valid_batch)):
                                    q = torch.from_numpy(y_valid_batch[ii]).to(device)
                                    w = torch.from_numpy(x_valid_batch[ii]).to(device)
                                    a = torch.from_numpy(aux_valid_batch[ii]).to(device)
                                    a = a.unsqueeze(1)
                                    yy.append(q)

                                    a = a.repeat(1, w.shape[1], 1)
                                    # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                                    # print('x_batch[:,5,0]',x_batch[:,5,0])
                                    # x_batch =torch.Tensor(w)
                                    w = torch.cat([w, a], 2)
                                    xx.append(w)
                                with torch.no_grad():
                                    pred_valid = model(xx,aux_valid_batch)


                                loss_weight = []
                                for ini in range(cfg['num_repeat']):
                                    loss_ = NaNMSELoss.fit(cfg, pred_valid[ini].squeeze().float(), yy[ini].squeeze().float(), lossmse)
                                    loss_weight.append(loss_)
                                weight = F.softmax(torch.randn(cfg['num_repeat']), dim=-1)
                                weight = weight.to(device)

                                # Random Loss Weighting  簡稱RLW
                                # loss 包含所有的損失 與隨機分配的權重張量相乘
                                # mse_valid_loss = sum(loss_weight[i] * weight[i] for i in range(cfg['num_repeat']))
                                mse_valid_loss = 0.2 * loss_weight[0] + 0.8 * loss_weight[1]
                                # for i in range(len(pred)):
                                #     loss += NaNMSELoss.fit(cfg, pred[i].float(), y_batch[i].float(), lossmse)

                                # 交叉熵
                                # mse_valid_loss2 = Cross.fit(cfg, pred_valid2.squeeze(1), y_valid_batch_v,lossmse_cross)
                                MSE_valid_loss += mse_valid_loss.item()
                                print('after_mse_valid_loss', mse_valid_loss)
                        #         测试用模型
                        if cfg["modelname"] in ['FAMLSTM',"EDLSTM"]:#FAMLSTM,REDFLSTM
                            gt_list = [i for i in range(0,x_valid[0].shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid[0].shape[0]-cfg["seq_len"])//cfg["stride"]

                            losses_val = []

                            for i in range(0, n):
                                #mask
                                x_valid_batch,y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn_nomal(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n , mask_index_val)
                                yy = []
                                xx = []
                                pred_valid = []
                                for ii in range(len(y_valid_batch)):
                                    q = torch.from_numpy(y_valid_batch[ii]).to(device)
                                    w = torch.from_numpy(x_valid_batch[ii]).to(device)
                                    a = torch.from_numpy(aux_valid_batch[ii]).to(device)
                                    a = a.unsqueeze(1)
                                    yy.append(q)

                                    a = a.repeat(1, w.shape[1], 1)
                                    # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                                    # print('x_batch[:,5,0]',x_batch[:,5,0])
                                    # x_batch =torch.Tensor(w)
                                    w = torch.cat([w, a], 2)
                                    xx.append(w)
                                with torch.no_grad():
                                    pred_valid = model(xx,aux_valid_batch,cfg)


                                loss_weight = []
                                for ini in range(cfg['num_repeat']):
                                    loss_ = NaNMSELoss.fit(cfg, pred_valid[ini].squeeze().float(), yy[ini].squeeze().float(), lossmse)
                                    loss_weight.append(loss_)
                                weight = F.softmax(torch.randn(cfg['num_repeat']), dim=-1)
                                weight = weight.to(device)

                                # Random Loss Weighting  簡稱RLW
                                # loss 包含所有的損失 與隨機分配的權重張量相乘
                                # mse_valid_loss = sum(loss_weight[i] * weight[i] for i in range(cfg['num_repeat']))
                                resu = sum(loss_weight)
                                mse_valid_loss =resu/cfg['num_repeat']
                                # for i in range(len(pred)):
                                #     loss += NaNMSELoss.fit(cfg, pred[i].float(), y_batch[i].float(), lossmse)

                                # 交叉熵
                                # mse_valid_loss2 = Cross.fit(cfg, pred_valid2.squeeze(1), y_valid_batch_v,lossmse_cross)
                                MSE_valid_loss += mse_valid_loss.item()
                                print('after_mse_valid_loss', mse_valid_loss)

                        elif cfg['modelname'] in ['MTLCNN']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network
                                for num in range(cfg['num_repeat']):
                                    x_valid_batch[num][np.isnan(x_valid_batch[num])] = 0
                                    x_valid_batch[num] = torch.Tensor(x_valid_batch[num]).to(device)
                                    y_valid_batch[num] = torch.Tensor(y_valid_batch[num]).to(device)
                                    aux_valid_batch[num] = torch.Tensor(aux_valid_batch[num]).to(device)
                                    # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                    x_valid_batch[num] = x_valid_batch[num].squeeze(1)
                                    x_valid_batch[num] = x_valid_batch[num].reshape(x_valid_batch[num].shape[0],x_valid_batch[num].shape[1]*x_valid_batch[num].shape[2],x_valid_batch[num].shape[3],x_valid_batch[num].shape[4])
                                    x_valid_batch[num] = torch.cat([x_valid_batch[num], aux_valid_batch[num]], axis=1)

                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                losses= []
                                loss_sum = 0.0
                                for i_val in range(cfg['num_repeat']):
                                    mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid[i_val].squeeze(dim=1).float(), y_valid_batch[i_val].float(),
                                                               lossmse)
                                    loss_sum += mse_valid_loss
                                    losses.append(mse_valid_loss)
                                mse_valid_loss = loss_sum / cfg['num_repeat']
                                MSE_valid_loss += mse_valid_loss.item()
                        elif cfg['modelname'] in ['CrossStitchNet']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_CS(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network
                                for num in range(cfg['num_repeat']):
                                    x_valid_batch[num][np.isnan(x_valid_batch[num])] = 0
                                    x_valid_batch[num] = torch.Tensor(x_valid_batch[num]).to(device)
                                    y_valid_batch[num] = torch.Tensor(y_valid_batch[num]).to(device)
                                    aux_valid_batch[num] = torch.Tensor(aux_valid_batch[num]).to(device)
                                    # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                    x_valid_batch[num] = x_valid_batch[num].squeeze(1)
                                    x_valid_batch[num] = x_valid_batch[num].reshape(x_valid_batch[num].shape[0],x_valid_batch[num].shape[1]*x_valid_batch[num].shape[2],x_valid_batch[num].shape[3],x_valid_batch[num].shape[4])
                                    x_valid_batch[num] = torch.cat([x_valid_batch[num], aux_valid_batch[num]], axis=1)

                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                losses= []
                                loss_sum = 0.0
                                for i_val in range(cfg['num_repeat']):
                                    mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid[i_val].squeeze(dim=1).float(), y_valid_batch[i_val].float(),
                                                               lossmse)
                                    loss_sum += mse_valid_loss
                                    losses.append(mse_valid_loss)
                                mse_valid_loss = loss_sum / cfg['num_repeat']
                                MSE_valid_loss += mse_valid_loss.item()
                        elif cfg['modelname'] in ['MTANmodel']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time']-cfg["multi_step"],cfg["stride"])]
                            n = (x_valid[0].shape[0]-cfg["seq_len"])//cfg["stride"]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:

                                x_cnn, y_cnn, aux_cnn, _, _,mask_cnn = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"],mask_index) # same as Convolutional Neural Network
                                x_lstm, y_lstm, aux_lstm = \
                                    load_test_data_for_rnn1(cfg, x_valid_lstm, y_valid_lstm, static_valid_lstm,
                                                            scaler_y,
                                                            cfg["stride"], i, n,mask_cnn)
                                for pre_num in range(cfg['num_repeat']):
                                    # load cnn test data
                                    x_cnn[pre_num][np.isnan(x_cnn[pre_num])] = 0  # filter nan values to train cnn model
                                    x_cnn[pre_num] = torch.from_numpy(x_cnn[pre_num]).to(device)
                                    aux_cnn[pre_num] = torch.from_numpy(aux_cnn[pre_num]).to(device)
                                    y_cnn[pre_num] = torch.from_numpy(y_cnn[pre_num]).to(device)
                                    # x_cnn[pre_num] = x_cnn[pre_num].squeeze(dim=1)

                                    x_cnn[pre_num] = x_cnn[pre_num].reshape(x_cnn[pre_num].shape[0],
                                                                            x_cnn[pre_num].shape[1] * x_cnn[pre_num].shape[
                                                                                2], x_cnn[pre_num].shape[3],
                                                                            x_cnn[pre_num].shape[4])
                                    #  for all day
                                    # aux_cnn[pre_num] = aux_cnn[pre_num].unsqueeze(1)
                                    # aux_cnn[pre_num] = aux_cnn[pre_num].repeat(1, x_cnn[pre_num].shape[1], 1, 1, 1)
                                    # x_cnn[pre_num] = torch.cat([x_cnn[pre_num], aux_cnn[pre_num]], 2)
                                    x_cnn[pre_num] = torch.cat([x_cnn[pre_num], aux_cnn[pre_num]], 1)
                                    #  load lstm test data
                                    y_lstm[pre_num] = torch.from_numpy(y_lstm[pre_num]).to(device)
                                    x_lstm[pre_num] = torch.from_numpy(x_lstm[pre_num]).to(device)
                                    aux_lstm[pre_num] = torch.from_numpy(aux_lstm[pre_num]).to(device)
                                    aux_lstm[pre_num] = aux_lstm[pre_num].unsqueeze(1)

                                    aux_lstm[pre_num] = aux_lstm[pre_num].repeat(1, x_lstm[pre_num].shape[1], 1)

                                    x_lstm[pre_num] = torch.cat([x_lstm[pre_num], aux_lstm[pre_num]], 2)
                                    # 直接使用 x_lstm[pre_num] 的前两个维度，并设置最后一个维度为 128
                                    zeros_tensor = torch.zeros(x_lstm[pre_num].size(0), x_lstm[pre_num].size(1), 128).to(device)

                                with torch.no_grad():
                                    # predictions_val = []
                                    # # 由于x_cnn,x_lstm是list  进去model之后会影响model本来的值
                                    # input_data = x_lstm.copy()
                                    # new_input = []
                                    # for step in range(cfg["multi_step"]):
                                    #     pred = model(input_data, aux_lstm, cfg)
                                    #     step_pred = pred.copy()
                                    #     predictions_val.append(step_pred)
                                    #     for pre_num in range(cfg['num_repeat']):
                                    #         # 将预测结果添加到输入序列中作为一个新的观察
                                    #         old = pred[pre_num]
                                    #         old = old.unsqueeze(-1)  # 或者 pred.unsqueeze(2)，结果相同
                                    #         # 然后，使用 expand 方法沿最后一个维度复制值，得到形状为 [128, 1, 14]
                                    #         old_expanded = old.expand(-1, -1, 14)
                                    #         new_input.append(torch.cat((input_data[pre_num], old_expanded), dim=1))
                                    #         input_data[pre_num] = new_input[pre_num][:, 1:, :]
                                    pred_valid = model(zeros_tensor,x_lstm, aux_cnn,cfg)
                                # losses1 = []
                                # losses2 = []
                                # loss_sum = 0.0
                                # optim.zero_grad()
                                # for step in range(cfg["multi_step"]):
                                #     for idex_Tass in range(cfg['num_repeat']):
                                #         pred_ = predictions_val[step][idex_Tass].squeeze().float()
                                #         loss_time = NaNMSELoss.fit(cfg, pred_, y_lstm[idex_Tass][:, step].float(), lossmse)
                                #         if idex_Tass == 0:
                                #             losses1.append(loss_time)
                                #         else:
                                #             losses2.append(loss_time)
                                #         loss_sum = loss_sum + loss_time
                                    for i_val in range(cfg['num_repeat']):
                                        mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid[i_val].squeeze().float(), y_lstm[i_val][:,0].float(),
                                                                   lossmse)
                                        mse_valid_loss = torch.mul(model.weights[i_val], mse_valid_loss)
                                        loss_sum += mse_valid_loss
                                        losses.append(mse_valid_loss)
                                    mse_valid_loss = loss_sum / cfg['num_repeat']
                                    MSE_valid_loss += mse_valid_loss.item()

                        elif cfg['modelname'] in ['MTLConvLSTMModel']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len']-cfg['forcast_time'],cfg["stride"])]
                            valid_batch_size = cfg["batch_size"]*10
                            for i in gt_list:
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y,gt_list,lat_index,lon_index, i ,cfg["stride"]) # same as Convolutional Neural Network
                                for num in range(cfg['num_repeat']):
                                    x_valid_batch[num][np.isnan(x_valid_batch[num])] = 0
                                    x_valid_batch[num] = torch.Tensor(x_valid_batch[num]).to(device)
                                    y_valid_batch[num] = torch.Tensor(y_valid_batch[num]).to(device)
                                    aux_valid_batch[num] = torch.Tensor(aux_valid_batch[num]).to(device)
                                    # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
                                    aux_valid_batch[num] = aux_valid_batch[num].unsqueeze(1)
                                    aux_valid_batch[num] = aux_valid_batch[num].repeat(1, x_valid_batch[num].shape[1], 1, 1, 1)

                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch,cfg)
                                losses= []
                                loss_sum = 0.0
                                for i_val in range(cfg['num_repeat']):
                                    mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid[i_val].squeeze(dim=1).float(), y_valid_batch[i_val].float(),
                                                               lossmse)
                                    loss_sum += mse_valid_loss
                                    losses.append(mse_valid_loss)
                                mse_valid_loss = loss_sum / cfg['num_repeat']
                                MSE_valid_loss += mse_valid_loss.item()
                        if cfg["modelname"] in ['MMOE']:
                            gt_list = [i for i in range(0,x_valid[0].shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid[0].shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                x_valid_batch,y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn1(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                yy = []
                                xx = []
                                pred_valid = []
                                for i in range(len(y_valid_batch)):
                                    y_valid_batch[i] = torch.from_numpy(y_valid_batch[i]).to(device)
                                    w = torch.from_numpy(x_valid_batch[i]).to(device)
                                    a = torch.from_numpy(aux_valid_batch[i]).to(device)
                                    a = a.unsqueeze(1)

                                    a = a.repeat(1, w.shape[1], 1)
                                    # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                                    # print('x_batch[:,5,0]',x_batch[:,5,0])
                                    # x_batch =torch.Tensor(w)
                                    w = torch.cat([w, a], 2)
                                    xx.append(w)
                                with torch.no_grad():
                                    pred_valid = model(xx,aux_valid_batch)
                                losses = []
                                loss_sum = 0.0
                                for i_val in range(cfg['num_repeat']):
                                    mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid[i_val].squeeze(dim=1).float(),
                                                                    y_valid_batch[i_val].float(),
                                                                    lossmse)
                                    loss_sum += mse_valid_loss
                                    losses.append(mse_valid_loss)
                                mse_valid_loss = loss_sum / cfg['num_repeat']
                                MSE_valid_loss += mse_valid_loss.item()
                        if cfg["modelname"] in ['LSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                if isinstance(y_valid, list):
                                    y_valid = y_valid[0]
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()


#  validate way for CNN model
# ------------------------------------------------------------------------------------------------------------------------------


                        if cfg["modelname"] in ['MSLSTMModel','SoftMTLv1','MMOE','MTLCNN','MTLConvLSTMModel','CrossStitchNet','REDFLSTM',"MTANmodel","FAMLSTM","EDLSTM"]:
                            t_end = time.time()
                            print('gt_list=',gt_list)
                            print('gt_list.len=',len(gt_list))
                            mse_valid_loss = MSE_valid_loss / (len(gt_list))
                            print(mse_valid_loss)
                            val_save_acc = mse_valid_loss
                        # if cfg["modelname"] in ['MTANmodel']:
                        #
                        #
                        #     mse_valid_loss = MSE_valid_loss / (len(gt_list))
                        #     print(mse_valid_loss)
                        #     val_save_acc = mse_valid_loss

                        else:
                            t_end = time.time()
                            mse_valid_loss = MSE_valid_loss/(len(gt_list))
                            # get loss log
                            loss_str = '\033[1;31m%s\033[0m' % \
                                    "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch,mse_valid_loss,
                                        t_end-t_begin)
                            print(loss_str)
                            val_save_acc = mse_valid_loss

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc < best:
                        #if MSE_valid_loss < best:
                            if cfg["modelname"] in ['SoftMTLv1']:
                                torch.save(model1,out_path+cfg['modelname']+'_para.pkl')
                                torch.save(model2,out_path+'SoftMTLv2'+'_para.pkl')
                                wait = 0  # release wait
                                best = val_save_acc #MSE_valid_loss
                                print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                            else:
                                torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                                wait = 0  # release wait
                                best = val_save_acc #MSE_valid_loss
                                print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                        else:
                            print('it is wait {}'.format(wait))
                else:
                    if cfg["modelname"] in ['SoftMTLv1']:
                        if MSELoss < best:
                            best = MSELoss
                            wait = 0
                            torch.save(model1, out_path + cfg['modelname'] + '_para.pkl')
                            torch.save(model2, out_path + 'SoftMTLv2' + '_para.pkl')
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                    # save best model by train loss
                    if MSELoss < best:
                        best = MSELoss
                        wait = 0
                        torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                # early stopping
                if wait >= patience:
                    # print(epoch_losses_sum[0].shape)
                    # _plotloss(cfg,epoch_losses_sum)
                    # _plotbox(cfg,epoch_losses_sum)
                    # _boxkge(cfg,kge)
                    # _boxpcc(cfg,pcc)
                    # _boxpcc_data(cfg,pcc_data)
                    # _boxnse(cfg,nse)
                    return
            return


