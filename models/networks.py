import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

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

    
class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, activation=True, normalization=True, activation_name='LeakyReLU'):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        if normalization:
            # FC block doesn't support InstanceNorm1d
            if isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm1d:
                norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU
        if activation:
            if activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc_block(x)
        return y

        
class AENet(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_sizes, latent_size, split_B, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=2048, dim_2A=1024, dim_1C=1024, dim_2C=1024, dim_3=512):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                input_sizes (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_size (int)        -- the dimensionality of the latent space
        """

        super().__init__()

        self.A_dim = input_sizes[0]
        self.B_dim_list = input_sizes[1]
        self.C_dim = input_sizes[2]
        self.dim_1B = dim_1B

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1A = FCBlock(self.A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 3
        self.encode_fc_3B = FCBlock(dim_2B, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_3A = FCBlock(dim_2A, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                        activation=False, normalization=False)
        self.encode_fc_3C = FCBlock(dim_2C, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                        activation=False, normalization=False)   

        # DECODER
        # Layer 1
        self.decode_fc_3B = FCBlock(latent_size, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)
        self.decode_fc_3A = FCBlock(latent_size, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)
        self.decode_fc_3C = FCBlock(latent_size, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)

        # Layer 2
        self.decode_fc_2B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_2A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_2C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 3
        self.decode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_1B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))
        self.decode_fc_1A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_1C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)


    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)
        level_2_A = self.encode_fc_1A(x[0])
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        
        h_B = self.encode_fc_3B(level_3_B)
        h_A = self.encode_fc_3A(level_3_A)
        h_C = self.encode_fc_3C(level_3_C)

        return h_A, h_B, h_C

    def decode(self, h):
        level_3_B = self.decode_fc_3B(h[1])
        level_3_A = self.decode_fc_3A(h[0])
        level_3_C = self.decode_fc_3C(h[2])

        level_2_B = self.decode_fc_2B(level_3_B)
        level_2_A = self.decode_fc_2A(level_3_A)
        level_2_C = self.decode_fc_2C(level_3_C)

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_1B_list[i](level_2_B.narrow(1, i * self.dim_1B, self.dim_1B)))
        recon_A = self.decode_fc_1A(level_2_A)
        recon_C = self.decode_fc_1C(level_2_C)

        return [recon_A, recon_B_list, recon_C]

    def forward(self, x):
        return self.encode(x)