'''
Title: Custum MLP for GearNet
Author: Gijs Tukkers
Date: 10-07-2023
'''
import torch
from torch import nn
import torch.nn.functional as F

class Reduce_to_20_concat_one_hot(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(Reduce_to_20_concat_one_hot, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*9*hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Reduce_to_20(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(Reduce_to_20, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, hidden1, bias=False),
            nn.Dropout(p=0.5)
            )

    def forward(self, x):
        return self.model(x)


class MLP_after_encoder(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(MLP_after_encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2*hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9*input_shape, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Autoencoder_linear(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(Autoencoder_linear, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9*input_shape, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden1, 9*input_shape)
        )

    def forward(self, x):
        return self.model(x)


class Reduce_1_hidden_layer(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(Reduce_1_hidden_layer, self).__init__()

        self.model = nn.Sequential(
            # nn.Linear(input_shape, hidden1, bias=False),
            # nn.Dropout(p=0.5),
            # nn.Flatten(),
            nn.Linear(9 * 2* hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
class MLP_1_hidden_layer(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(MLP_1_hidden_layer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, hidden1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
class Conv_1_hidden_layer(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(Conv_1_hidden_layer, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(input_shape, hidden1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.permute(0,2,1))
    
class Conv2_1_hidden_layer(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(Conv2_1_hidden_layer, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(input_shape, hidden1, 1),
            nn.ReLU(hidden1),
            nn.Conv1d(hidden1, hidden2, 1),
            nn.ReLU(hidden2),
            nn.Flatten(),
            nn.Linear(9 * hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.permute(0,2,1))


class MLP_reduce_input(nn.Module):
    def __init__(self, output=1, input_shape=None, hidden1=None, hidden2=None, hidden3=None, dropout_rate=None):
        super(MLP_reduce_input, self).__init__()
        #input
        self.mlp_reduce = nn.Linear(input_shape, hidden1, bias=False)

    def forward(self,x):
        return self.mlp_reduce(x)