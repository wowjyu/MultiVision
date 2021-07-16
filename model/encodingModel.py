import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ChartTypeNN(nn.Module):
    def __init__(self, feature_size, num_class):
        super().__init__()
        self.feature_size = feature_size
        self.num_class = num_class
        
        self.dense = nn.Sequential()
        layers = []
        linearLayers = [self.feature_size, 256, 128, 64, 32]
        dropout_rate = 0.2
        for i in range((len(linearLayers)-1)):
            self.dense.add_module("linear{0}".format(i), nn.Linear(linearLayers[i], linearLayers[i+1]))
            self.dense.add_module("rulu{0}".format(i), nn.LeakyReLU()) 
            self.dense.add_module("dropout{0}".format(i),nn.Dropout(dropout_rate))
            
        self.dense.add_module("linear{0}".format(len(linearLayers)), nn.Linear(linearLayers[-1], num_class))
        self.dense.add_module("output", nn.ReLU())

    def forward(self, x):
        x = x.view(-1, self.feature_size)
        out =  self.dense(x)
        return out


class ChartTypeLSTM(nn.Module):
    """Score Net"""
    def __init__(self, input_size, seq_length, num_class, bidirectional=False, hidden_size=200, pack = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.seq_length = seq_length
        
        self.pack = pack
        number_direction = 2 if bidirectional else 1

        self.lstm = nn.LSTM(bidirectional=bidirectional, input_size = input_size, hidden_size = hidden_size, num_layers = 1, batch_first=True)

#         self.linear = nn.Linear(hidden_size*seq_length*number_direction, num_class)
        #  self.relu = nn.ReLU()

        self.dense = nn.Sequential()
        layers = []
        linearLayers = [hidden_size*seq_length*number_direction, 2000, 1600, 1200]
        dropout_rate = 0.4
        for i in range((len(linearLayers)-1)):
            self.dense.add_module("linear{0}".format(i), nn.Linear(linearLayers[i], linearLayers[i+1]))
            self.dense.add_module("rulu{0}".format(i), nn.LeakyReLU()) 
            self.dense.add_module("dropout{0}".format(i),nn.Dropout(dropout_rate))
            
        self.dense.add_module("linear{0}".format(len(linearLayers)), nn.Linear(linearLayers[-1], num_class))
        self.dense.add_module("output", nn.ReLU())     

    def forward(self, input_seq):
        (all_outs, (final_oupt,final_state)) = self.lstm(input_seq) 
          
        ## todo: 1024
        all_outs = all_outs.contiguous().view(all_outs.shape[0], -1)
        
        return self.dense(all_outs)
#         print(all_outs.shape)
#         oupt = self.linear(all_outs)
                
#         return self.relu(oupt)

class ScoreNetLSTM(nn.Module):
    """Score Net"""
    def __init__(self, input_size, seq_length, bidirectional=False, hidden_size=200, output_size=1, batch_size=1024, pack = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.seq_length = seq_length
        
        self.pack = pack
        number_direction = 2 if bidirectional else 1

        self.lstm = nn.LSTM(bidirectional=bidirectional, input_size = input_size, hidden_size = hidden_size, num_layers = 1, batch_first=True)
        self.linear = nn.Linear(hidden_size*seq_length*number_direction, output_size)

        
    def forward(self, input_seq):
        (all_outs, (final_oupt,final_state)) = self.lstm(input_seq) 
#        when not pack: print(all_outs.shape) ## batch_size * seq_length * hidden_size
    
        if self.pack == True:
            all_outs, _ = pad_packed_sequence(all_outs, batch_first=True) 
#             print(all_outs.shape)
            
        ## todo: 1024
        all_outs = all_outs.contiguous().view(all_outs.shape[0], -1)
#         print(all_outs.shape)
        oupt = self.linear(all_outs)
        
        return oupt