import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, input_size, output_size, drop_p):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(input_size // 2, input_size // 3),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(input_size // 3, output_size)
        )

    def forward(self, x):
        return self.layers(x)
    

class Decoder(torch.nn.Module):
    def __init__(self, input_size, output_size, drop_p):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size // 3),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(output_size // 3, output_size // 2),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(output_size // 2, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x