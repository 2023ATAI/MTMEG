import numpy as np
import torch
from utils import r2_score
import time
from data_gen import erath_data_transform
import sys
from data import Dataset

def batcher_lstm(x_test, y_test, aux_test, seq_len,forcast_time):
    n_t, n_feat = x_test.shape
    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, seq_len, n_feat))*np.nan
    # print('x_new',x_new.shape)
    y_new = np.zeros((n,2))*np.nan
    # print('y_new', y_new.shape)
    aux_new = np.zeros((n,aux_test.shape[0]))*np.nan
   #x_new每次加载7天 y_new加载8天 多处一天是每个7天进行一次预测？
    for i in range(n):
        x_new[i] = x_test[i:i+seq_len]
        y_new[i] = y_test[i+seq_len+forcast_time]
        aux_new[i] = aux_test
    # print("x_new  is", x_new)
    return x_new, y_new, aux_new

def batcher_mtl(x_test, y_test, aux_test, seq_len,forcast_time):
    a = torch.from_numpy(x_test)
    n_t, n_feat = x_test.shape
    _, n_out = y_test.shape
    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, seq_len, n_feat))*np.nan
    # print('x_new',x_new.shape)
    y_new = np.zeros((n,n_out))*np.nan
    # print('y_new', y_new.shape)
    aux_new = np.zeros((n,aux_test.shape[0]))*np.nan
   #x_new每次加载7天 y_new加载8天 多处一天是每个7天进行一次预测？
    for i in range(n):
        x_new[i] = x_test[i:i+seq_len]
        #shape 是 365，2
        y_new[i] = y_test[i+seq_len+forcast_time]

        aux_new[i] = aux_test
    # print("x_new  is", x_new)
    # shape 是 365，1
    return x_new, y_new,  aux_new

def batcher_convlstm(x_test, y_test, aux_test, seq_len,forcast_time,spatial_offset,i,j,lat_index,lon_index):
    x_test = x_test.transpose(0,3,1,2)
    y_test = y_test.transpose(0,3,1,2)
    aux_test = aux_test.transpose(2, 0, 1)

    n_t, n_feat, n_lat,n_lon = x_test.shape

    nt = y_test.shape[1]
    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, 1, n_feat,2*spatial_offset+1,2*spatial_offset+1))*np.nan
    y_new = np.zeros((n,1))*np.nan


    aux_new = np.zeros((n, aux_test.shape[0], 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan
    for ni in range(n):
        lat_index_bias = lat_index[i] + spatial_offset
        lon_index_bias = lon_index[j] + spatial_offset
        x_new[ni] = x_test[ni+seq_len:ni+seq_len+1,:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
        y_new[ni] = y_test[ni+seq_len+forcast_time,:,i,j]
        # 这一行会出错  1.26  第一次跑会错 第二次就能过
        aux_new[ni] = aux_test[:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]

    return x_new, y_new, aux_new


def test_mtl(x, y, static, scaler, cfg, model,device):
    cls = Dataset(cfg)
    # 蒙特卡罗 Dropout(Monte-Carlo Dropout)
    model.eval()
    # model.eval()
    if cfg['modelname'] in ['MTLCNN','MTLConvLSTMModel','MTANmodel','CrossStitchNet']:
        #	Splice x according to the sphere shape
        lat_index, lon_index = erath_data_transform(cfg, x)
        print(
            '\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(
                m_n=cfg['modelname']))
    y_pred_enses = []
    all_day = []
    y_truees = []
    mask = []
    #y_pred_ens = np.zeros((y.shape[0] - cfg["seq_len"] - cfg['forcast_time'], y.shape[1], y.shape[2], 2, 7)) * np.nan

    for i in range(cfg['num_repeat']):
        y_pred_ens = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
        y_true = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,i]
        y_pred_enses.append(y_pred_ens)
        y_truees.append(y_true)
        y_true = np.array(y_true)
        mask_time =y_true == y_true
        mask.append(mask_time)

    print('x shape is',x.shape)
    print('y_true shape is',y.shape)

    t_begin = time.time()
    # ------------------------------------------------------------------------------------------------------------------------------
    # for each grid by MTL model
    if cfg["modelname"] in ['MTLCNN']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"], cfg['forcast_time'],
                                                            cfg["spatial_offset"], i, j, lat_index, lon_index)
                for num in range(cfg['num_repeat']):
                    x_new[num] = np.nan_to_num(x_new[num])
                    static_new[num] = np.nan_to_num(static_new[num])
                    x_new[num] = torch.from_numpy(x_new[num]).to(device)
                    static_new[num] = torch.from_numpy(static_new[num]).to(device)
                    # x_new = torch.cat([x_new, static_new], 1)
                    x_new[num] = x_new[num].squeeze(1)
                    x_new[num] = x_new[num].reshape(x_new[num].shape[0], x_new[num].shape[1] * x_new[num].shape[2], x_new[num].shape[3], x_new[num].shape[4])
                    x_new[num] = torch.cat([x_new[num], static_new[num]], 1)
                pred = model(x_new, static_new)
                pp = []
                for ii in range(len(pred)):
                    p = pred[ii].cpu().detach().numpy()
                    p = np.squeeze(p)
                    if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                        p = cls.reverse_normalize(p, 'output', scaler[:, i, j, ii], 'minmax', -1)
                    elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                        p = cls.reverse_normalize(p, 'output', scaler, 'minmax', -1)
                    pp.append(p)
                # print('pre reverse is',pred.shape)
                # print('pred2 reverse is',pred2)
                for iq in range(cfg['num_repeat']):
                    y_pred_enses[iq][:,i,j] = pp[iq]
                # y_pred_ens1[:, i, j] = pp[0]
                # y_pred_ens2[:, i, j] = pp[1]
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1
    if cfg["modelname"] in ['CrossStitchNet']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_cnn = []
                x_lstm = []
                y_cnn = []
                y_lstm = []
                static_cnn = []
                for num in range(cfg['num_repeat']):
                    x_new, y_new, static_new = batcher_convlstm(x, y[:,:,:,num:num+1], static[num], cfg["seq_len"], cfg['forcast_time'],
                                                                cfg["spatial_offset"], i, j, lat_index, lon_index)

                    x_new = np.nan_to_num(x_new)
                    static_new = np.nan_to_num(static_new)
                    x_new = torch.from_numpy(x_new).to(device)
                    static_new = torch.from_numpy(static_new).to(device)
                    # x_new = torch.cat([x_new, static_new], 1)
                    x_new = x_new.squeeze(1)
                    x_new = x_new.reshape(x_new.shape[0], x_new.shape[1] * x_new.shape[2], x_new.shape[3], x_new.shape[4])
                    x_new = torch.cat([x_new, static_new], 1)
                    x_cnn.append(x_new)
                    y_cnn.append(y_new)
                    static_cnn.append(static_new)
                pred = model(x_cnn, static_cnn)
                pp = []
                for ii in range(len(pred)):
                    p = pred[ii].cpu().detach().numpy()
                    p = np.squeeze(p)
                    if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                        p = cls.reverse_normalize(p, 'output', scaler[:, i, j, ii], 'minmax', -1)
                    elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                        p = cls.reverse_normalize(p, 'output', scaler, 'minmax', -1)
                    pp.append(p)
                # print('pre reverse is',pred.shape)
                # print('pred2 reverse is',pred2)
                for iq in range(cfg['num_repeat']):
                    y_pred_enses[iq][:,i,j] = pp[iq]
                # y_pred_ens1[:, i, j] = pp[0]
                # y_pred_ens2[:, i, j] = pp[1]
                if count % 1000== 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1
    if cfg["modelname"] in ['MTANmodel']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_cnn = []
                x_lstm = []
                y_cnn = []
                y_lstm = []
                static_cnn = []

                for num in range(cfg['num_repeat']):
                    x_new, y_new, static_new = batcher_convlstm(x, y[:,:,:,num:num+1], static[num], cfg["seq_len"], cfg['forcast_time'],
                                                                cfg["spatial_offset"], i, j, lat_index, lon_index)

                    x_new = np.nan_to_num(x_new)
                    static_new = np.nan_to_num(static_new)
                    x_new = torch.from_numpy(x_new).to(device)
                    static_new = torch.from_numpy(static_new).to(device)
                    # x_new = torch.cat([x_new, static_new], 1)
                    # x_new = x_new.squeeze(1)
                    # for one day
                    x_new = x_new.reshape(x_new.shape[0], x_new.shape[1] * x_new.shape[2], x_new.shape[3], x_new.shape[4])
                    x_new = torch.cat([x_new, static_new], 1)
                    # for all day
                    # static_new = static_new.unsqueeze(1)
                    # static_new = static_new.repeat(1, x_new.shape[1], 1, 1, 1)
                    # x_new = torch.cat([x_new, static_new], 2)

                    x_cnn.append(x_new)
                    y_cnn.append(y_new)
                    static_cnn.append(static_new)



                for num in range(cfg['num_repeat']):
                    x_new, y_new, static_new = batcher_mtl(x[:, i, j, :], y[:, i, j, :], static[num][i, j, :], cfg["seq_len"],
                                                            cfg['forcast_time'])

                    xt = torch.from_numpy(x_new).to(device)
                    s = torch.from_numpy(static_new).to(device)
                    s = s.unsqueeze(1)
                    s = s.repeat(1, xt.shape[1], 1)
                    xt = torch.cat([xt, s], 2)
                    x_lstm.append(xt)


                input_data = x_lstm.copy()
                new_input = []
                predictions = []

                for step in range(cfg["multi_step"]):
                    pred = model(input_data, static_cnn,cfg)
                    pp = []
                    for ii in range(len(pred)):
                        p = pred[ii].cpu().detach().numpy()
                        p = np.squeeze(p)
                        if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                            p = cls.reverse_normalize(p, 'output', scaler[:, i, j, ii], 'minmax', -1)
                        elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                            p = cls.reverse_normalize(p, 'output', scaler, 'minmax', -1)
                        pp.append(p)
                    step_pred = pp.copy()
                    predictions.append(step_pred)
                    for pre_num in range(cfg['num_repeat']):
                        # 将预测结果添加到输入序列中作为一个新的观察
                        old = pred[pre_num]
                        old = old.unsqueeze(-1)  # 或者 pred.unsqueeze(2)，结果相同
                        # 然后，使用 expand 方法沿最后一个维度复制值，得到形状为 [128, 1, 14]
                        old_expanded = old.expand(-1, -1, 14)
                        new_input.append(torch.cat((input_data[pre_num], old_expanded), dim=1))
                        input_data[pre_num] = new_input[pre_num][:, 1:, :]
                # print('pre reverse is',pred.shape)
                # print('pred2 reverse is',pred2)
                    for iq in range(cfg['num_repeat']):
                        y_pred_ens[:,i,j,iq,step] = step_pred[iq]
                    # y_pred_ens1[:, i, j] = pp[0]
                    # y_pred_ens2[:, i, j] = pp[1]
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1

    if cfg["modelname"] in ['MTLConvLSTMModel']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"], cfg['forcast_time'],
                                                            cfg["spatial_offset"], i, j, lat_index, lon_index)
                for num in range(cfg['num_repeat']):
                    x_new[num] = np.nan_to_num(x_new[num])
                    static_new[num] = np.nan_to_num(static_new[num])
                    x_new[num] = torch.from_numpy(x_new[num]).to(device)
                    static_new[num] = torch.from_numpy(static_new[num]).to(device)
                    # x_new = torch.cat([x_new, static_new], 1)
                    static_new[num]  = static_new[num] .unsqueeze(1)
                    static_new[num]  = static_new[num] .repeat(1, x_new[num] .shape[1], 1, 1, 1)
                pred = model(x_new, static_new,cfg)
                pp = []
                for ii in range(len(pred)):
                    p = pred[ii].cpu().detach().numpy()
                    p = np.squeeze(p)
                    if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                        p = cls.reverse_normalize(p, 'output', scaler[:, i, j, ii], 'minmax', -1)
                    elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                        p = cls.reverse_normalize(p, 'output', scaler, 'minmax', -1)
                    pp.append(p)
                # print('pre reverse is',pred.shape)
                # print('pred2 reverse is',pred2)
                for iq in range(cfg['num_repeat']):
                    y_pred_enses[iq][:,i,j] = pp[iq]
                # y_pred_ens1[:, i, j] = pp[0]
                # y_pred_ens2[:, i, j] = pp[1]
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1

    if cfg["modelname"] in ['MSLSTMModel','MMOE','REDFLSTM','FAMLSTM',"EDLSTM"]:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                xx = []
                for num in range(cfg['num_repeat']):
                    x_new, y_new, static_new = batcher_mtl(x[:, i, j, :], y[:, i, j, :], static[num][i, j, :], cfg["seq_len"],
                                                            cfg['forcast_time'])

                    xt = torch.from_numpy(x_new).to(device)
                    s = torch.from_numpy(static_new).to(device)
                    s = s.unsqueeze(1)
                    s = s.repeat(1, xt.shape[1], 1)
                    xt = torch.cat([xt, s], 2)
                    xx.append(xt)
                pred = model(xx, static_new,cfg)
#
                # pred = pred*std[i, j, :]+mean[i, j, :] #(nsample,1,1)

                # print('pred1  is', pred.shape)
                # print('pred2  is', pred2)
                #經過 反正則化出來的結果就是nan
                pp = []
                for ii in range(len(pred)):
                    p = pred[ii].cpu().detach().numpy()
                    p = np.squeeze(p)
                    if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                        p = cls.reverse_normalize(p, 'output', scaler[:, i, j, ii], 'minmax', -1)
                    elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                        p = cls.reverse_normalize(p, 'output', scaler, 'minmax', -1)
                    pp.append(p)
                # print('pre reverse is',pred.shape)
                # print('pred2 reverse is',pred2)
                for iq in range(cfg['num_repeat']):
                    y_pred_enses[iq][:,i,j] = pp[iq]
                # y_pred_ens1[:, i, j] = pp[0]
                # y_pred_ens2[:, i, j] = pp[1]
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1
# ------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

    #  7月5  預測數據y_pred_ens1和y_pred_ens2是空的
    t_end = time.time()
    print('y_pred_ens shape is',y_pred_ens.shape)


    print('scaler shape is',scaler.shape)

    # 多步预测
    # return y_pred_ens,y_truees
    return y_pred_enses,y_truees


