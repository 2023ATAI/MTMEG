

import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTM
from Transformer import  TransformerLayer,TransformerEncoder
from TransformerLayer_task import TransformerLayer_task
# ------------------------------------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        return x + out
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)  
        return output, (hidden, cell)  

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, Rinput, encoder_hidden):
       
        output, _ = self.lstm(Rinput, encoder_hidden)
        return output

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_vector = nn.Parameter(torch.rand(hidden_size, 1))

    def forward(self, lstm_output):
        attention_weights = torch.tanh(lstm_output) @ self.attention_vector
        attention_weights = torch.softmax(attention_weights.squeeze(2), dim=1)
        context_vector = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights



class REDFLSTM(nn.Module):
    """Enhanced Deep Time-Series LSTM model with Feed-Forward Attention"""

    def __init__(self, cfg, lstmmodel_cfg):
        super(REDFLSTM, self).__init__()
        self.encoder = Encoder(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"])
        self.decoder = Decoder(lstmmodel_cfg["hidden_size"], 7)
        self.lstm1 = nn.LSTM(lstmmodel_cfg["hidden_size"] + 1, lstmmodel_cfg["hidden_size"], batch_first=True)
        self.attention = Attention(lstmmodel_cfg["hidden_size"])  # Attention layer
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
        # Adding a fully connected layer with ReLU activation
        self.fc = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["hidden_size"])
        self.relu = nn.ReLU()
        self.head_layers = nn.ModuleList()
        self.dense1 = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
        self.dense2 = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
        self.head_layers.append(self.dense1)
        self.head_layers.append(self.dense2)

        self.change_size = nn.Linear(14, cfg["hidden_size"])
        self.share_TF = TransformerEncoder(num_layers=2, inputsize=128, hidden_dim=cfg["hidden_size"], num_heads=2,
                                           pf_dim=cfg["hidden_size"] * 2)
        self.task_TF = TransformerLayer_task(128, hidden_dim=cfg["hidden_size"], num_heads=2,
                                             pf_dim=cfg["hidden_size"] * 2)
        self.qureys_SA = share_qureySA(cfg["hidden_size"] * cfg["num_repeat"])
        self.task_SA = share_qureySA(cfg["hidden_size"])

    def forward(self, inputs, aux, cfg):
        inputs_ = inputs.copy()
        pred = []
        share_features = []
        for i in range(2):
            # MTL-TF
            tf_in = self.change_size(inputs_[i])
            x_tf, q_share = self.share_TF(cfg, tf_in.float())
            share_features.append(self.drop(x_tf))
        # combine features
        share_features = torch.cat(share_features, dim=2)
        share_features = self.qureys_SA(share_features)
        for i in range(2):
            # Encoder
            encoder_hidden = self.encoder(inputs_[i].float())
            # Decoder
            seq_len = inputs[i].size(1)  # Sequence length for decoder output
            decoder_output = self.decoder(encoder_hidden, seq_len)
            decoder_output = decoder_output.permute(1, 2, 0)

            task_features = share_features[:, :, i * cfg['hidden_size']:(i + 1) * cfg['hidden_size']]

            inputs_[i] = self.change_size(inputs_[i])
            inputs_[i] = self.task_SA(inputs_[i])
            x_task = self.task_TF(cfg, inputs_[i].float(), task_features.float()) + inputs_[i]

            # Combine encoder output with decoder output and the auxiliary input
            combined = torch.cat((decoder_output, x_task.float()), dim=2)
            # Pass combined inputs through second LSTM layer
            lstm_output, _ = self.lstm1(combined.float())

            # Apply attention mechanism to the output of the second LSTM
            context_vector, _ = self.attention(lstm_output)

            # Apply dropout and pass through the dense layer
            aa = self.drop(context_vector)

            # Applying ReLU activation function
            aa = self.relu(self.fc(aa))
            # aa = aa[:,-1,:]

            outputs = self.head_layers[i](aa)
            pred.append(outputs)

        return pred

class share_qureySA(nn.Module):
    def __init__(self, in_channels):
        super(share_qureySA, self).__init__()
        self.query_conv = nn.Linear(in_channels, in_channels)
        self.key_conv = nn.Linear(in_channels, in_channels)
        self.value_conv = nn.Linear(in_channels, in_channels)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, seq, inputdim = x.size()

        # Project features to query, key, and value
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        dk = proj_key.size(-1)

        # Compute attention scores
        # attention = torch.relu(torch.matmul(proj_query.permute(0, 2, 1), proj_key))
        attention = torch.nn.functional.softmax(torch.matmul(proj_query, proj_key.permute(0, 2, 1)), dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, proj_value)

        # Scale and add residual connection
        out = out + x

        return out

        
class Expert(nn.Module):
    def __init__(self,cfg):
        super(Expert, self).__init__()

        self.task_TF = TransformerLayer_task(128,hidden_dim=cfg["hidden_size"],num_heads=2,pf_dim=cfg["hidden_size"]*2)
        self.mlp_task = MLP(cfg["hidden_size"])
        self.change_size1 = nn.Linear(15,cfg["hidden_size"])

    def forward(self, cfg,x,task_features):

        x = self.change_size1(x.float())
        x_task = self.task_TF(cfg, x.float(), task_features.float())

        return x_task
import torch.nn.functional as F
class Gate(nn.Module):
    def __init__(self, input_size, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_size, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)
        
        
class FAMLSTM(nn.Module):
        # """Enhanced Deep Time-Series LSTM model with Feed-Forward Attention"""

    def __init__(self, cfg, lstmmodel_cfg):
        super(FAMLSTM, self).__init__()
        self.lstm1 = nn.LSTM(128, lstmmodel_cfg["hidden_size"], batch_first=True)
        self.lstm2 = nn.LSTM(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["hidden_size"], batch_first=True)
        self.attention = Attention(lstmmodel_cfg["hidden_size"])  # Attention layer
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
        # Adding a fully connected layer with ReLU activation
        self.fc = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["hidden_size"])
        self.relu = nn.ReLU()
        self.head_layers = nn.ModuleList()
        #self.dense1 = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
        #self.dense2 = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
        #self.head_layers.append(self.dense1)
        #self.head_layers.append(self.dense2)       
        self.head_layers = nn.ModuleList([
            nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"]) 
            for _ in range(cfg["num_repeat"])
        ])
        self.change_size1 = nn.Linear(15,cfg["hidden_size"])
        self.change_size_back = nn.Linear(cfg["hidden_size"],15)
        self.share_TF = TransformerEncoder(num_layers=2,inputsize=128,hidden_dim=cfg["hidden_size"],num_heads=2,pf_dim=cfg["hidden_size"]*2)
        self.task_TF = TransformerLayer_task(128,hidden_dim=cfg["hidden_size"],num_heads=2,pf_dim=cfg["hidden_size"]*2)
        self.qureys_SA = share_qureySA(cfg["hidden_size"] * cfg["num_repeat"])
        self.task_SA = share_qureySA(cfg["hidden_size"])
        self.mlp_task = MLP(cfg["hidden_size"])
        self.mlp_share = MLP(cfg["hidden_size"] * 2)
        self.experts = Expert(cfg) 
        self.gates = nn.ModuleList([Gate(cfg["hidden_size"],cfg["num_repeat"]) for _ in range(cfg["num_repeat"])]) # 定义两个门控
    def forward(self, inputs,aux,cfg):
        inputs_ = inputs.copy()
        pred = []
        share_features = []
        mid_output = []
        for i in range(5):
            # MTL-TF
            tf_in = self.change_size1(inputs_[i].float())
            x_tf, q_share = self.share_TF(cfg,tf_in.float())
            share_features.append(self.drop(x_tf))
        # combine features
        share_features = torch.cat(share_features,dim=2)
        #share_features = self.mlp_share(share_features.float())
        share_features = self.qureys_SA(share_features)
        Task_F = []
        for i in range(5):
            expert_outputs = self.experts(cfg,inputs_[i],share_features)
            x = self.change_size1(inputs_[i].float())


           # Combine encoder output with decoder output and the auxiliary input

            # Pass combined inputs through second LSTM layer
            lstm_output1, _ = self.lstm1(expert_outputs.float())


            #lstm_output2, _ = self.lstm2(lstm_output1.float())
            # Apply attention mechanism to the output of the second LSTM

            # Apply dropout and pass through the dense layer

            # Applying ReLU activation function
            # aa = aa[:,-1,:]
            outputs = self.head_layers[i](lstm_output1[:,-1,:])
            pred.append(outputs)

        return pred

class EDLSTM(nn.Module):
    # """Single task LSTM model with attention mechanism."""

    def __init__(self, cfg, lstmmodel_cfg):
        super(EDLSTM, self).__init__()
        self.encoder1 = Encoder(lstmmodel_cfg["input_size"],
                               lstmmodel_cfg["hidden_size"])


        self.decoder1 = Decoder(lstmmodel_cfg["hidden_size"],
                               lstmmodel_cfg["hidden_size"])
                               
        self.lstm = nn.LSTM(
            input_size=lstmmodel_cfg["input_size"],
            hidden_size=lstmmodel_cfg["hidden_size"],
            batch_first=True
        )
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
        self.attention_vector = nn.Parameter(torch.rand(lstmmodel_cfg["hidden_size"], 1))
        self.change_size1 = nn.Linear(15, cfg["hidden_size"])
        self.share_TF = TransformerEncoder(num_layers=2,inputsize=128,hidden_dim=cfg["hidden_size"],num_heads=2,pf_dim=cfg["hidden_size"]*2)
        self.task_TF = TransformerLayer_task(128,hidden_dim=cfg["hidden_size"],num_heads=2,pf_dim=cfg["hidden_size"]*2)
        self.qureys_SA = share_qureySA(cfg["hidden_size"] * cfg["num_repeat"])
        self.task_SA = share_qureySA(cfg["hidden_size"])
        self.residual_x = nn.Linear(15, cfg["hidden_size"])
        self.head_layers = nn.ModuleList()
        self.mlp_task = MLP(cfg["hidden_size"])
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg["num_repeat"])])
        self.gates = nn.ModuleList([Gate(cfg["hidden_size"]) for _ in range(cfg["num_repeat"])]) 
        for i in range(cfg['num_repeat']):
            self.head_layers.append(nn.Linear(128,1))
    def forward(self, inputs, aux,cfg):
        pred = []
        q_shares = []
        inputs_ = inputs.copy()
        mid_output = []
        for i in range(cfg['num_repeat']):
            tf_input = self.change_size1(inputs_[i].float())
            x_tf, q_share = self.share_TF(cfg, tf_input.float())
            # x_tf = self.mpl(x_tf)
            q_shares.append(self.drop(x_tf))

        share_features = torch.cat(q_shares,dim=2)
        share_features = self.qureys_SA(share_features)
        # q_share = self.qureys_SA_LN(q_share)
        # q_share = self.mpl(q_share)
        for i in range(2):
            expert_outputs = [expert(cfg,inputs_[j],share_features) for j, expert in enumerate(self.experts)]
            x = self.change_size1(inputs_[i].float())
            gate_output = self.gates[i](x)
            output = torch.zeros_like(expert_outputs[0])
            for expert_output in expert_outputs:
                output += gate_output * expert_output

            _, (hidden, cell) = self.encoder1(inputs_[i].float())
            outputs = self.decoder1(output, (hidden, cell))

            # Apply dropout to the context vector
            context_vector = self.drop(outputs)

            # Pass the context vector through the dense layer to get final output
            output = self.head_layers[i](context_vector[:, -1, :])
            pred.append(output)

        return pred
