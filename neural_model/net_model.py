import torch.nn as nn
import torch
from torch.autograd import Variable

class netModel(nn.Module):

    def __init__(self, input_size):
        super(netModel, self).__init__()
        self.first_layer = nn.Sequential(nn.Linear(input_size, 160),
                                         nn.ReLU(inplace=True))
        self.hidden_layers = nn.Sequential(nn.Dropout(p=0.3),
                                           nn.Linear(160, 240),
                                           nn.ReLU(inplace=True))
        self.final_layer = nn.Linear(240, 1)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.final_layer(x)
        return x


class netModel_I(nn.Module):

    def __init__(self, input_size):
        super(netModel_I, self).__init__()
        self.first_layer = nn.Sequential(nn.Linear(input_size, 32),
                                         nn.ReLU(inplace=True))
        self.hidden_layers = nn.Sequential(nn.Dropout(p=0.3),
                                           nn.Linear(32, 48),
                                           nn.ReLU(inplace=True)
                                           )
        self.final_layer = nn.Linear(48, 1)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.final_layer(x)
        return x


class model_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(model_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.first_layer = nn.Sequential(nn.Linear(hidden_size, 40),
                                         nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)
        self.hidden_layers = nn.Sequential(nn.Dropout(p=0.3),
                                           nn.Linear(40, 100),
                                           nn.ReLU(inplace=True),
                                           )
        self.final_layer = nn.Linear(100, 1)

    def forward(self, x):
        #x = torch.cat(x, 1)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm_layer(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.first_layer(out)
        out = self.hidden_layers(out)
        out = self.final_layer(out)
        return out
