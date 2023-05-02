import torch.nn as nn
import torch
import math as m
import torch.nn.functional as F
import math

class EMGhandnet(nn.Module):
    def __init__(self):
        super(EMGhandnet, self).__init__()
        self.cnn = CNNmodel()
        self.lstm = Bi_LSTMModel()
        
        self.fc1 = nn.Linear(10000,512)
        self.fcc = nn.Linear(64,128)
        self.fcc2 =nn.Linear(128,64)
        self.fc2 = nn.Linear(512,52)
    def forward(self,x,x2):
        #input (batch, 25,10, 20)
        temp = [ self.cnn(x[:,t,:,:]) for t in range(x.size(1))]
        """
        for t in range(x.size(1)):
            temp.append(self.cnn(x[:,t,:,:]))
        """
        x = torch.stack(temp,1)
        temp = [ self.cnn(x2[:,t,:,:]) for t in range(x2.size(1))]
        x2 = torch.stack(temp,1)
        # (batch,time, features) = (batch, 25, 64)
        
        x = self.lstm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.2)
        x = F.relu(self.fc2(x))
        
        return x
    
class Bi_LSTMModel(nn.Module):
    def __init__(self):
        super(Bi_LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = 200

        # Number of hidden layers
        self.num_layers = 2

        # batch_first=True causes input/output tensors to be of shape
        # (z, seq_dim, feature_dim) (25,10,200)
        # (seq_length, batch, feature dim) (25, 32, 64)
        self.lstm = nn.LSTM(64, self.hidden_dim, self.num_layers, dropout=0.2093,bidirectional=True, batch_first=True)
        #self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, dropout=0.2093,bidirectional=True, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool1d(20)

    def forward(self, x):
        
        #print(x.shape)
        # input = (25, batch, 64)
        
        
        # Initialize hidden state with zeros
        # (4,batch,200)
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_dim,device=x.device).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_dim,device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #print(cn)
        
        #(batch, 25, 400)
        #print(out.shape)
        
        #out = self.avgpool(out)
        out = torch.flatten(out,1)
        ##print(out.shape)
        #print('e')
        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        #out = self.fc(out[:, -1, :]).squeeze()
        # out.size() --> 100, 10
        return out
    
class CNNmodel(nn.Module):
    def __init__(self,stride=2, padding_mode='zeros'):
        super(CNNmodel,self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=64, kernel_size=9, 
                  stride=stride, padding=4, padding_mode='zeros')
        torch.nn.init.normal_(self.conv1.weight, mean = 0, std = m.sqrt(1/32))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=stride, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv2.weight, mean = 0, std = m.sqrt(1/32))
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, 
                  stride=stride, padding=2, padding_mode='zeros')
        torch.nn.init.normal_(self.conv3.weight, mean = 0, std = m.sqrt(1/32))
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, 
                  stride=stride, padding=1, padding_mode='zeros')
        torch.nn.init.normal_(self.conv4.weight, mean = 0, std = m.sqrt(1/32))
        
        self.norm1 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        self.norm2 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        self.norm3 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        self.norm4 = nn.BatchNorm1d(64,eps=1e-6,momentum=0.05)
        
        self.maxpool = nn.MaxPool1d(kernel_size=8,stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(2)
        
    def forward(self,x):
        #input (batch, 10, 20) = (batch, emg_ch, unit_time)
        x = F.relu(self.conv1(x))
        # (batch, 64, 10) = (batch_size, num_filters, 10)
        
        x = self.maxpool(x)
        # (batch, 64,2)
        
        x = F.relu(self.conv2(x))
        # (batch, 64, 1)
        
        x = F.relu(self.conv3(x))
        # (batch, 64, 1)
        
        x = F.relu(self.conv4(x))
        # (batch, 64, 1)
        
        x= torch.flatten(x,1)
        # (batch, 64)
        
        
        return x    