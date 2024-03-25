import numpy as np
import pandas as pd
import math
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp.autocast_mode as autocast
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from seq_load import *

class CNN200_RNN(nn.Module):
    def __init__(self , HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT, FC_DROPOUT, CELL):
        super(CNN51_RNN ,self).__init__()
        self.basicconv0a = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(1, 7), stride=(1,2), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.basicconv0b = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), stride=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.basicconv2a = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.rnn = BiLSTM_Attention(128 ,HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT)
        self.fc1 = nn.Linear(HIDDEN_NUM * 2, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(FC_DROPOUT)

    def forward(self, x):
        x = x.unsqueeze(3).permute(0, 2, 3, 1)
        x = self.basicconv0a(x)
        x = self.basicconv0b(x)
        x = self.basicconv2a(x) 
        x = x.squeeze(2).permute(2, 0, 1)  
        out  = self.rnn(x) 
        out = self.dropout(self.fc1(out))
        out = F.relu(out)
        out = self.fc2(out)
        return out


class BiLSTM_Attention(nn.Module):
    def __init__(self ,input_size, HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT):
        super(BiLSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM, bidirectional=True, dropout=RNN_DROPOUT)
    def attention_net(self, lstm_output, final_state ):
        HIDDEN_NUM = 128
        hidden = final_state.view(-1, HIDDEN_NUM * 2, 3) 
        hidden = torch.mean(hidden, 2).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) 
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context 

    def forward(self, x):
        input = x
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2) 
        attn_output= self.attention_net(output, final_hidden_state)
        return attn_output

