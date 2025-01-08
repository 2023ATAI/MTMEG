"""
All model structure

<<<<<<<< HEAD
Author: Qingliang Li 
12/23/2022 - V1.0  LSTM, CNN, ConvLSTM edited by Qingliang Li
.........  - V2.0
.........  - V3.0
"""

import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTM


class LSTMModel(nn.Module):
    """single task model"""

    def __init__(self, cfg,lstmmodel_cfg):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"],batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"],lstmmodel_cfg["out_size"])

    def forward(self, inputs,aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:])
        return x

class CNN(nn.Module):
    """single task model"""

    def __init__(self, cfg):
        super(CNN,self).__init__()
        self.latn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.lonn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.cnn = nn.Conv2d(in_channels=cfg["input_size_cnn"],out_channels=cfg["hidden_size"],kernel_size=cfg["kernel_size"],stride=cfg["stride_cnn"])
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1)

    def forward(self, inputs,aux):
        x = self.cnn(inputs.float())
        x = self.drop(x)
        x = x.reshape(x.shape[0],-1)
        # we only predict the last step
        x = self.dense(x)
        return x
# 硬共享
class MTLCNN(nn.Module):
    """double task model"""

    def __init__(self, cfg):
        super(MTLCNN, self).__init__()
        # 3 3 2
        self.latn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1  # 3
        self.lonn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1  # 3
        # 68 128  3 2
        self.cnn = nn.Conv2d(in_channels=cfg["input_size_cnn"],out_channels=cfg["hidden_size"],kernel_size=cfg["kernel_size"],stride=cfg["stride_cnn"])
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        # self.dense = nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1)

        self.head_layers = nn.ModuleList()
        for i in range(cfg['num_repeat']):
            self.head_layers.append(nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1))

    # 多個數據輸入
    def forward(self, inputs,aux):
        pred = []
        for i in range(len(self.head_layers)):
            x = self.cnn(inputs[i].float())
            x = self.drop(x)
            x = x.reshape(x.shape[0], -1)
            pred.append(self.head_layers[i](x))
        return pred

class MTLConvLSTMModel(nn.Module):


    def __init__(self, cfg):
        super(MTLConvLSTMModel,self).__init__()
        self.ConvLSTM_net = ConvLSTM(input_size=(int(2*cfg["spatial_offset"]+1),int(2*cfg["spatial_offset"]+1)),
                       input_dim=int(cfg["input_size"]),
                       hidden_dim=[int(cfg["hidden_size"]), int(cfg["hidden_size"]/2)],
                       kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
                       num_layers=2,cfg=cfg,batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        # self.dense = nn.Linear(int(cfg["hidden_size"]/2)*int(2*cfg["spatial_offset"]+1)*int(2*cfg["spatial_offset"]+1),1)
        #self.batchnorm = nn.BatchNorm1d(int(cfg["hidden_size"]/2))
        self.head_layers = nn.ModuleList()
        for i in range(cfg['num_repeat']):
            self.head_layers.append(nn.Linear(int(cfg["hidden_size"]/2)*int(2*cfg["spatial_offset"]+1)*int(2*cfg["spatial_offset"]+1),1))

    def forward(self, inputs,aux,cfg):
        pred = []
        for i in range(len(self.head_layers)):
            threshold = torch.nn.Threshold(0., 0.0)
            inputs_new = torch.cat([inputs[i], aux[i]], 2).float()
            #inputs_new = inputs.float()
            hidden =  self.ConvLSTM_net.get_init_states(inputs_new.shape[0])
            last_state, encoder_state =  self.ConvLSTM_net(inputs_new.clone(), hidden)
            last_state = self.drop(last_state)
            #Convout = last_state[:,-1,:,cfg["spatial_offset"],cfg["spatial_offset"]]
            Convout = last_state[:,-1,:,:,:]
            #Convout = self.batchnorm(Convout)
            shape=Convout.shape[0]
            #print('Convout shape is',Convout.shape)
            Convout=Convout.reshape(shape,-1)
            Convout = torch.flatten(Convout,1)
            Convout = threshold(Convout)
            predictions=self.head_layers[i](Convout)
            pred.append(predictions)
        return pred
# one exper
class aMMOE(nn.Module):
    def __init__(self, cfg, MMOE_cfg):
        super(MMOE, self).__init__()
        self.expert0 = nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True)
            # nn.Dropout(p=cfg["dropout_rate"])


        self.gate0 = nn.Sequential(
            nn.Linear(MMOE_cfg["input_size"], 1),
            nn.Softmax(dim=1)
        )
        self.gate1 = nn.Sequential(
            nn.Linear(MMOE_cfg["input_size"], 1),
            nn.Softmax(dim=1)
        )

        self.Gates = nn.ModuleList([
            self.gate0,self.gate1
        ])


        self.tower0 = nn.Linear(MMOE_cfg["hidden_size"], MMOE_cfg["out_size"])


        self.tower1 = nn.Linear(MMOE_cfg["hidden_size"], MMOE_cfg["out_size"])

        self.Towers = nn.ModuleList([
            self.tower0,self.tower1
        ])

    def forward(self, inputs_, aux):

        gate_weights = []
        task_outputs = []
        combined_outputs = []
        for j,inputs in enumerate(inputs_):
            expert_output, _ = self.expert0(inputs.float())


            gate_model = self.Gates[j]
            gate_weight = gate_model(inputs.float())

                # 原本是128  7 3  和  128  7 128  不能相乘  转置后变成128 3 7 就可以相乘了
            combined_output = torch.matmul(gate_weight.transpose(1,2), expert_output)
            combined_outputs.append(combined_output)

            # task_outputs = [Towers(combined_output) for Towers in self.Towers]
        for i,tower_model in enumerate(self.Towers):
            output = tower_model(combined_outputs[i][:,-1:])
            task_outputs.append(output)

        return task_outputs
# 多專家
class GateModule(nn.Module):
    def __init__(self, input_size):
        super(GateModule, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x

class MMOE(nn.Module):
    def __init__(self, cfg, MMOE_cfg):
        super(MMOE, self).__init__()
        self.expert0 = nn.Sequential(
            nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True),
            # nn.Dropout(p=cfg["dropout_rate"])
        )
        self.expert1 = nn.Sequential(
            nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True),
            # nn.Dropout(p=cfg["dropout_rate"])
        )
        self.expert2 = nn.Sequential(
            nn.LSTM(MMOE_cfg["input_size"], MMOE_cfg["hidden_size"], batch_first=True),
            # nn.Dropout(p=cfg["dropout_rate"])
        )
        self.Expers = nn.ModuleList([
            self.expert0,self.expert1,self.expert2
        ])


        self.tower_layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        for i in range(cfg['num_repeat']):
            self.gates.append(GateModule(MMOE_cfg["input_size"]))
            self.tower_layers.append(nn.Linear(MMOE_cfg["hidden_size"],MMOE_cfg["out_size"]))

    def forward(self, inputs_, aux):

        gate_weights = []
        task_outputs = []
        combined_outputs = []
        for j,inputs in enumerate(inputs_):
            expert_outputs = []
            for expert_model in self.Expers:
                output,_ = expert_model(inputs.float())
                expert_outputs.append(output)

            gate_model = self.gates[j]
            gate_weight = gate_model(inputs.float())

            combined_output = 0
            for expert_output in expert_outputs:
                # 原本是128  7 3  和  128  7 128  不能相乘  转置后变成128 3 7 就可以相乘了
                combined_output += torch.matmul(gate_weight.transpose(1,2), expert_output)
            combined_outputs.append(combined_output)

            # task_outputs = [Towers(combined_output) for Towers in self.Towers]
        # for i,tower_model in enumerate(self.Towers):
            output = self.tower_layers[j](combined_outputs[j][:,-1:])
            task_outputs.append(output)

        return task_outputs


import torch.nn.functional as F

class CrossStitchUnit(nn.Module):
    def __init__(self):
        super(CrossStitchUnit, self).__init__()
        # Set weights as a trainable scalar parameter，0到1以内随机
        self.weight1 = nn.Parameter(torch.rand(1))
        self.weight2 = nn.Parameter(1-self.weight1)


    def forward(self, inputs):
        # Element-wise multiplication by the trainable scalar
        x1_cross = inputs[0] * self.weight1
        x2_cross = inputs[1] * self.weight2

        return x1_cross, x2_cross


class CrossStitchNet(nn.Module):
    def __init__(self, cfg):
        super(CrossStitchNet, self).__init__()

        self.latn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1  # 3
        self.lonn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1  # 3
        # 68 128  3 2
        self.cnn1 = nn.Conv2d(in_channels=cfg["input_size_cnn"],out_channels=cfg["hidden_size"],kernel_size=cfg["kernel_size"],stride=cfg["stride_cnn"])
        self.cnn2 = nn.Conv2d(in_channels=cfg["input_size_cnn"],out_channels=cfg["hidden_size"],kernel_size=cfg["kernel_size"],stride=cfg["stride_cnn"])
        self.drop1 = nn.Dropout(p=cfg["dropout_rate"])
        self.drop2 = nn.Dropout(p=cfg["dropout_rate"])
        # self.dense = nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1)

        self.head_layers = nn.ModuleList()
        for i in range(cfg['num_repeat']):
            self.head_layers.append(nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1))

        # CrossStitch Units
        self.cross_stitch_unit1 = CrossStitchUnit()
        self.cross_stitch_unit2 = CrossStitchUnit()

    def forward(self, inputs,aux):
        pred = []
        # task 1
        x1 = self.cnn1(inputs[0].float())
        x1 = self.drop1(x1)
        x1 = x1.reshape(x1.shape[0], -1)
        pred.append(self.head_layers[0](x1))
        # task 2
        x2 = self.cnn2(inputs[1].float())
        x2 = self.drop2(x2)
        x2 = x2.reshape(x2.shape[0], -1)
        pred.append(self.head_layers[1](x2))

        # CrossStitch Units
        task1_x_cross1, task2_x_cross1 = self.cross_stitch_unit1(pred)
        task1_x_cross2, task2_x_cross2 = self.cross_stitch_unit2(pred)

        # Combine the tasks using the cross-stitched features
        pred[0] = task1_x_cross1 + task2_x_cross1
        pred[1] = task2_x_cross2 + task1_x_cross2

        return pred
