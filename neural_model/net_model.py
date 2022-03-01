import torch.nn as nn
import torch


class netModel(nn.Module):

    def __init__(self, input_size):
        super(netModel, self).__init__()
        self.first_layer = nn.Sequential(nn.Linear(input_size, 320),
                                         nn.ReLU(inplace=True))
        self.hidden_layers = nn.Sequential(nn.Dropout(p=0.5),
                                           nn.Linear(320, 250),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(250, 160),
                                           nn.ReLU(inplace=True))
        self.final_layer = nn.Linear(160, 1)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.final_layer(x)
        return x
