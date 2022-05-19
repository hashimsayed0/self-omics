import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import string

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


class SimCLRProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(SimCLRProjectionHead,self).__init__(**kwargs)
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

class CLIPProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
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

        
class AESepB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_sizes, latent_size, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=2048, dim_2A=1024, dim_1C=1024, dim_2C=1024):
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

class AESepA(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_sizes, latent_size, use_one_decoder, concat_latent_for_decoder, recon_all_thrice, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=1024, dim_2B=1024,
                 dim_1A=128, dim_2A=1024, dim_1C=1024, dim_2C=1024, dim_2 = 2048, dim_1=1024):
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

        self.A_dim_list = input_sizes[0]
        self.B_dim = input_sizes[1]
        self.C_dim = input_sizes[2]
        self.dim_1B = dim_1B
        self.dim_1A = dim_1A
        self.use_one_decoder = use_one_decoder
        self.concat_latent_for_decoder = concat_latent_for_decoder
        self.recon_all_thrice = recon_all_thrice

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(self.B_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_1A_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1A_list.append(
                FCBlock(self.A_dim_list[i], dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A*23, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
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
        if self.concat_latent_for_decoder:
            latent_size *= 3

        if self.use_one_decoder:
            self.decode_fc_3 = FCBlock(latent_size, dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)
            self.decode_fc_2 = FCBlock(dim_2, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
            self.decode_fc_1B = FCBlock(dim_1, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
            self.decode_fc_1A_list = nn.ModuleList()
            for i in range(0, 23):
                self.decode_fc_1A_list.append(
                    FCBlock(dim_1, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=False, normalization=False))
            self.decode_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                        activation=False, normalization=False)
            
        else:
            self.decode_fc_3B = FCBlock(latent_size, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=False, normalization=False)
            self.decode_fc_3A = FCBlock(latent_size, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)
            self.decode_fc_3C = FCBlock(latent_size, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)

            if self.recon_all_thrice:
                dim_1 = dim_1A * 23
                # Layer 2
                self.decode_fc_2B = FCBlock(dim_2B, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2A = FCBlock(dim_2A, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2C = FCBlock(dim_2C, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                
                # Layer 3
                self.decode_from_A_fc_1B = FCBlock(dim_1, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
                self.decode_from_A_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_A_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_A_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
                
                self.decode_from_B_fc_1B = FCBlock(dim_1, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
                self.decode_from_B_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_B_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_B_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)

                self.decode_from_C_fc_1B = FCBlock(dim_1, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
                self.decode_from_C_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_C_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_C_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)    

            else:
                # Layer 2
                self.decode_fc_2B = FCBlock(dim_2B, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2A = FCBlock(dim_2A, dim_1A*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
                # Layer 3
                self.decode_fc_1B = FCBlock(dim_1B, self.B_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
                self.decode_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_fc_1C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)


    def encode(self, x):
        level_2_B = self.encode_fc_1B(x[1])
        level_2_A_list = []
        for i in range(0, 23):
            level_2_A_list.append(self.encode_fc_1A_list[i](x[0][i]))
        level_2_A = torch.cat(level_2_A_list, 1)
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        
        h_B = self.encode_fc_3B(level_3_B)
        h_A = self.encode_fc_3A(level_3_A)
        h_C = self.encode_fc_3C(level_3_C)

        return h_A, h_B, h_C

    def decode(self, h):
        if self.use_one_decoder:
            if self.concat_latent_for_decoder:
                z = torch.cat((h[0], h[1], h[2]), 1)
                level_3 = self.decode_fc_3(z)
                level_2 = self.decode_fc_2(level_3)
                recon_B = self.decode_fc_1B(level_2)
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_fc_1A_list[i](level_2))
                recon_C = self.decode_fc_1C(level_2)
            else:
                level_3_A = self.decode_fc_3(h[0])
                level_3_B = self.decode_fc_3(h[1])
                level_3_C = self.decode_fc_3(h[2])

                level_2_A = self.decode_fc_2(level_3_A)
                level_2_B = self.decode_fc_2(level_3_B)
                level_2_C = self.decode_fc_2(level_3_C)

                if self.recon_all_thrice:
                    recon_all_list = []
                    for level_2 in [level_2_A, level_2_B, level_2_C]:
                        recon_B = self.decode_fc_1B(level_2)
                        recon_A_list = []
                        for i in range(0, 23):
                            recon_A_list.append(self.decode_fc_1A_list[i](level_2))
                        recon_C = self.decode_fc_1C(level_2)
                        recon_all_list.append((recon_A_list, recon_B, recon_C))
                    return recon_all_list
                else:
                    recon_B = self.decode_fc_1B(level_2_B)
                    recon_A_list = []
                    for i in range(0, 23):
                        recon_A_list.append(self.decode_fc_1A_list[i](level_2_A))
                    recon_C = self.decode_fc_1C(level_2_C)

        else:
            if self.concat_latent_for_decoder:
                z = torch.cat((h[0], h[1], h[2]), 1)
                level_3_B = self.decode_fc_3B(z)
                level_3_A = self.decode_fc_3A(z)
                level_3_C = self.decode_fc_3C(z)
            else:
                level_3_B = self.decode_fc_3B(h[1])
                level_3_A = self.decode_fc_3A(h[0])
                level_3_C = self.decode_fc_3C(h[2])

            level_2_B = self.decode_fc_2B(level_3_B)
            level_2_A = self.decode_fc_2A(level_3_A)
            level_2_C = self.decode_fc_2C(level_3_C)

            if self.recon_all_thrice:
                recon_all_list = []

                recon_B = self.decode_from_A_fc_1B(level_2_A)
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_from_A_fc_1A_list[i](level_2_A.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_from_A_fc_1C(level_2_A)
                recon_all_list.append((recon_A_list, recon_B, recon_C))

                recon_B = self.decode_from_B_fc_1B(level_2_B)
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_from_B_fc_1A_list[i](level_2_B.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_from_B_fc_1C(level_2_B)
                recon_all_list.append((recon_A_list, recon_B, recon_C))

                recon_B = self.decode_from_C_fc_1B(level_2_C)
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_from_C_fc_1A_list[i](level_2_C.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_from_C_fc_1C(level_2_C)
                recon_all_list.append((recon_A_list, recon_B, recon_C))

                return recon_all_list

            else:
                recon_B = self.decode_fc_1B(level_2_B)
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_fc_1A_list[i](level_2_A.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_fc_1C(level_2_C)

        return [recon_A_list, recon_B, recon_C]

    def forward(self, x):
        return self.encode(x)


class AESepAB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_sizes, latent_size, use_one_decoder, concat_latent_for_decoder, recon_all_thrice, use_rep_trick, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=128, dim_2A=1024, dim_1C=1024, dim_2C=1024, dim_2 = 2048, dim_1=1024):
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

        self.A_dim_list = input_sizes[0]
        self.B_dim_list = input_sizes[1]
        self.C_dim = input_sizes[2]
        self.dim_1B = dim_1B
        self.dim_1A = dim_1A
        self.use_one_decoder = use_one_decoder
        self.concat_latent_for_decoder = concat_latent_for_decoder
        self.recon_all_thrice = recon_all_thrice
        self.use_rep_trick = use_rep_trick

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1A_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1A_list.append(
                FCBlock(self.A_dim_list[i], dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A*23, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
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

        if self.use_rep_trick:
            # Layer 4
            self.encode_fc_B_mean = FCBlock(latent_size, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                        activation=False, normalization=False)
            self.encode_fc_B_log_var = FCBlock(latent_size, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
            self.encode_fc_A_mean = FCBlock(latent_size, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                        activation=False, normalization=False)
            self.encode_fc_A_log_var = FCBlock(latent_size, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
            self.encode_fc_C_mean = FCBlock(latent_size, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                        activation=False, normalization=False)
            self.encode_fc_C_log_var = FCBlock(latent_size, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
        # DECODER
        # Layer 1
        if self.concat_latent_for_decoder:
            latent_size *= 3

        if self.use_one_decoder:
            self.decode_fc_3 = FCBlock(latent_size, dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)
            self.decode_fc_2 = FCBlock(dim_2, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
            self.decode_fc_1B_list = nn.ModuleList()
            for i in range(0, 23):
                self.decode_fc_1B_list.append(
                    FCBlock(dim_1, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=False, normalization=False))
            self.decode_fc_1A_list = nn.ModuleList()
            for i in range(0, 23):
                self.decode_fc_1A_list.append(
                    FCBlock(dim_1, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                            activation=False, normalization=False))
            self.decode_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                        activation=False, normalization=False)
            
        else:
            self.decode_fc_3B = FCBlock(latent_size, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=False, normalization=False)
            self.decode_fc_3A = FCBlock(latent_size, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)
            self.decode_fc_3C = FCBlock(latent_size, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=False, normalization=False)

            if self.recon_all_thrice:
                dim_1 = dim_1B * 23
                # Layer 2
                self.decode_fc_2B = FCBlock(dim_2B, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2A = FCBlock(dim_2A, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2C = FCBlock(dim_2C, dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                
                # Layer 3
                self.decode_from_A_fc_1B_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_A_fc_1B_list.append(
                        FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_A_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_A_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_A_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)
                
                self.decode_from_B_fc_1B_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_B_fc_1B_list.append(
                        FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_B_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_B_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_B_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)

                self.decode_from_C_fc_1B_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_C_fc_1B_list.append(
                        FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_C_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_from_C_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_from_C_fc_1C = FCBlock(dim_1, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)    

            else:
                # Layer 2
                self.decode_fc_2B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2A = FCBlock(dim_2A, dim_1A*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                            activation=True)
                self.decode_fc_2C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                        activation=True)
                # Layer 3
                self.decode_fc_1B_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_fc_1B_list.append(
                        FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_fc_1A_list = nn.ModuleList()
                for i in range(0, 23):
                    self.decode_fc_1A_list.append(
                        FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                activation=False, normalization=False))
                self.decode_fc_1C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                            activation=False, normalization=False)


    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)
        level_2_A_list = []
        for i in range(0, 23):
            level_2_A_list.append(self.encode_fc_1A_list[i](x[0][i]))
        level_2_A = torch.cat(level_2_A_list, 1)
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        
        h_B = self.encode_fc_3B(level_3_B)
        h_A = self.encode_fc_3A(level_3_A)
        h_C = self.encode_fc_3C(level_3_C)
        
        if self.use_rep_trick:
            h_A_mean = self.encode_fc_A_mean(h_A)
            h_A_var = self.encode_fc_A_log_var(h_A)
            h_B_mean = self.encode_fc_B_mean(h_B)
            h_B_var = self.encode_fc_B_log_var(h_B)
            h_C_mean = self.encode_fc_C_mean(h_C)
            h_C_var = self.encode_fc_C_log_var(h_C)
            z_A = self.reparameterize(h_A_mean, h_A_var)
            z_B = self.reparameterize(h_B_mean, h_B_var)
            z_C = self.reparameterize(h_C_mean, h_C_var)
            return z_A, z_B, z_C
        else:
            return h_A, h_B, h_C
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, h):
        if self.use_one_decoder:
            if self.concat_latent_for_decoder:
                z = torch.cat((h[0], h[1], h[2]), 1)
                level_3 = self.decode_fc_3(z)
                level_2 = self.decode_fc_2(level_3)
                recon_B_list = []
                for i in range(0, 23):
                    recon_B_list.append(self.decode_fc_1B_list[i](level_2))
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_fc_1A_list[i](level_2))
                recon_C = self.decode_fc_1C(level_2)
            else:
                level_3_A = self.decode_fc_3(h[0])
                level_3_B = self.decode_fc_3(h[1])
                level_3_C = self.decode_fc_3(h[2])

                level_2_A = self.decode_fc_2(level_3_A)
                level_2_B = self.decode_fc_2(level_3_B)
                level_2_C = self.decode_fc_2(level_3_C)

                if self.recon_all_thrice:
                    recon_all_list = []
                    for level_2 in [level_2_A, level_2_B, level_2_C]:
                        recon_B_list = []
                        for i in range(0, 23):
                            recon_B_list.append(self.decode_fc_1B_list[i](level_2))
                        recon_A_list = []
                        for i in range(0, 23):
                            recon_A_list.append(self.decode_fc_1A_list[i](level_2))
                        recon_C = self.decode_fc_1C(level_2)
                        recon_all_list.append((recon_A_list, recon_B_list, recon_C))
                    return recon_all_list
                else:
                    recon_B_list = []
                    for i in range(0, 23):
                        recon_B_list.append(self.decode_fc_1B_list[i](level_2_B))
                    recon_A_list = []
                    for i in range(0, 23):
                        recon_A_list.append(self.decode_fc_1A_list[i](level_2_A))
                    recon_C = self.decode_fc_1C(level_2_C)

        else:
            if self.concat_latent_for_decoder:
                z = torch.cat((h[0], h[1], h[2]), 1)
                level_3_B = self.decode_fc_3B(z)
                level_3_A = self.decode_fc_3A(z)
                level_3_C = self.decode_fc_3C(z)
            else:
                level_3_B = self.decode_fc_3B(h[1])
                level_3_A = self.decode_fc_3A(h[0])
                level_3_C = self.decode_fc_3C(h[2])

            level_2_B = self.decode_fc_2B(level_3_B)
            level_2_A = self.decode_fc_2A(level_3_A)
            level_2_C = self.decode_fc_2C(level_3_C)

            if self.recon_all_thrice:
                recon_all_list = []

                recon_B_list = []
                for i in range(0, 23):
                    recon_B_list.append(self.decode_from_A_fc_1B_list[i](level_2_A.narrow(1, i * self.dim_1B, self.dim_1B)))
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_from_A_fc_1A_list[i](level_2_A.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_from_A_fc_1C(level_2_A)
                recon_all_list.append((recon_A_list, recon_B_list, recon_C))

                recon_B_list = []
                for i in range(0, 23):
                    recon_B_list.append(self.decode_from_B_fc_1B_list[i](level_2_B.narrow(1, i * self.dim_1B, self.dim_1B)))
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_from_B_fc_1A_list[i](level_2_B.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_from_B_fc_1C(level_2_B)
                recon_all_list.append((recon_A_list, recon_B_list, recon_C))

                recon_B_list = []
                for i in range(0, 23):
                    recon_B_list.append(self.decode_from_C_fc_1B_list[i](level_2_C.narrow(1, i * self.dim_1B, self.dim_1B)))
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_from_C_fc_1A_list[i](level_2_C.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_from_C_fc_1C(level_2_C)
                recon_all_list.append((recon_A_list, recon_B_list, recon_C))

                return recon_all_list

            else:
                recon_B_list = []
                for i in range(0, 23):
                    recon_B_list.append(self.decode_fc_1B_list[i](level_2_B.narrow(1, i * self.dim_1B, self.dim_1B)))
                recon_A_list = []
                for i in range(0, 23):
                    recon_A_list.append(self.decode_fc_1A_list[i](level_2_A.narrow(1, i * self.dim_1A, self.dim_1A)))
                recon_C = self.decode_fc_1C(level_2_C)

        return [recon_A_list, recon_B_list, recon_C]

    def forward(self, x):
        return self.encode(x)


class VAESepB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_sizes, latent_size, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=2048, dim_2A=1024, dim_1C=1024, dim_2C=1024, dim_3=512):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                input_sizes (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(VAESepB, self).__init__()

        self.A_dim = input_sizes[0]
        self.B_dim_list = input_sizes[1]
        self.C_dim = input_sizes[2]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_2A = dim_2A
        self.dim_2C = dim_2C

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
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A+dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_size, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A+dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))
        self.decode_fc_4A = FCBlock(dim_1A, self.A_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
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
        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)
        level_2_C = level_2.narrow(1, self.dim_2B+self.dim_2A, self.dim_2C)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_B_list = []
        for i in range(0, 23):
            level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))
        level_3_A = self.decode_fc_3A(level_2_A)
        level_3_C = self.decode_fc_3C(level_2_C)

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))
        recon_A = self.decode_fc_4A(level_3_A)
        recon_C = self.decode_fc_4C(level_3_C)

        return [recon_A, recon_B_list, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class VAESepAB(nn.Module):
    """
        Defines a fully-connected variational autoencoder for multi-omics dataset
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_sizes, latent_size, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_1A=128, dim_2A=1024, dim_1C=1024, dim_2C=1024, dim_3=512):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                input_sizes (list)       -- the list of input omics dimensions
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """

        super(VAESepAB, self).__init__()

        self.A_dim_list = input_sizes[0]
        self.B_dim_list = input_sizes[1]
        self.C_dim = input_sizes[2]
        self.dim_1B = dim_1B
        self.dim_2B = dim_2B
        self.dim_1A = dim_1A
        self.dim_2A = dim_2A
        self.dim_2C = dim_2C

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(self.B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1A_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1A_list.append(
                FCBlock(self.A_dim_list[i], dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1C = FCBlock(self.C_dim, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A*23, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2C = FCBlock(dim_1C, dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B+dim_2A+dim_2C, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                      activation=False, normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3, latent_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                         activation=False, normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_size, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3, dim_2B+dim_2A+dim_2C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)

        # Layer 3
        self.decode_fc_3B = FCBlock(dim_2B, dim_1B*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3A = FCBlock(dim_2A, dim_1A*23, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.decode_fc_3C = FCBlock(dim_2C, dim_1C, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)

        # Layer 4
        self.decode_fc_4B_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4B_list.append(
                FCBlock(dim_1B, self.B_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))
        self.decode_fc_4A_list = nn.ModuleList()
        for i in range(0, 23):
            self.decode_fc_4A_list.append(
                FCBlock(dim_1A, self.A_dim_list[i], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                        activation=False, normalization=False))
        self.decode_fc_4C = FCBlock(dim_1C, self.C_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                    activation=False, normalization=False)

    def encode(self, x):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](x[1][i]))
        level_2_B = torch.cat(level_2_B_list, 1)
        level_2_A_list = []
        for i in range(0, 23):
            level_2_A_list.append(self.encode_fc_1A_list[i](x[0][i]))
        level_2_A = torch.cat(level_2_A_list, 1)
        level_2_C = self.encode_fc_1C(x[2])

        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3_C = self.encode_fc_2C(level_2_C)
        level_3 = torch.cat((level_3_B, level_3_A, level_3_C), 1)

        level_4 = self.encode_fc_3(level_3)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)
        level_2_B = level_2.narrow(1, 0, self.dim_2B)
        level_2_A = level_2.narrow(1, self.dim_2B, self.dim_2A)
        level_2_C = level_2.narrow(1, self.dim_2B+self.dim_2A, self.dim_2C)

        level_3_B = self.decode_fc_3B(level_2_B)
        level_3_B_list = []
        for i in range(0, 23):
            level_3_B_list.append(level_3_B.narrow(1, self.dim_1B*i, self.dim_1B))
        level_3_A = self.decode_fc_3A(level_2_A)
        level_3_A_list = []
        for i in range(0, 23):
            level_3_A_list.append(level_3_A.narrow(1, self.dim_1A*i, self.dim_1A))
        level_3_C = self.decode_fc_3C(level_2_C)

        recon_B_list = []
        for i in range(0, 23):
            recon_B_list.append(self.decode_fc_4B_list[i](level_3_B_list[i]))
        recon_A_list = []
        for i in range(0, 23):
            recon_A_list.append(self.decode_fc_4A_list[i](level_3_A_list[i]))
        recon_C = self.decode_fc_4C(level_3_C)

        return [recon_A_list, recon_B_list, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class ClassifierNet(nn.Module):
    """
    Defines a multi-layer fully-connected classifier
    """
    def __init__(self, class_num=2, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 class_dim_1=128, class_dim_2=64, layer_num=3):
        """
        Construct a multi-layer fully-connected classifier
        Parameters:
            class_num (int)         -- the number of class
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(ClassifierNet, self).__init__()

        self.input_fc = FCBlock(latent_dim, class_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True)

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = class_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, class_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout_p=dropout_flag*dropout_p, activation=True)]
            input_dim = class_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(class_dim_2, class_num, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


class SurvivalNet(nn.Module):
    """
    Defines a multi-layer fully-connected survival predictor
    """
    def __init__(self, time_num=256, latent_dim=128, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0,
                 down_dim_1=512, down_dim_2=256, layer_num=3):
        """
        Construct a multi-layer fully-connected survival predictor
        Parameters:
            time_num (int)          -- the number of time intervals in the model
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(SurvivalNet, self).__init__()

        self.input_fc = FCBlock(latent_dim, down_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True, activation_name='Tanh')

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = down_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, down_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout_p=dropout_p, activation=True, activation_name='Tanh')]
            input_dim = down_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        # the output dimension should be the number of time intervals
        self.output_fc = FCBlock(down_dim_2, time_num, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


class RegressionNet(nn.Module):
    """
    Defines a multi-layer fully-connected regression net
    """
    def __init__(self, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, down_dim_1=128,
                 down_dim_2=64, layer_num=3):
        """
        Construct a one dimensional multi-layer regression net
        Parameters:
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(RegressionNet, self).__init__()

        self.input_fc = FCBlock(latent_dim, down_dim_1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                activation=True)

        # create a list to store fc blocks
        mul_fc_block = []
        # the block number of the multi-layer fully-connected block should be at least 3
        block_layer_num = max(layer_num, 3)
        input_dim = down_dim_1
        dropout_flag = True
        for num in range(0, block_layer_num-2):
            mul_fc_block += [FCBlock(input_dim, down_dim_2, norm_layer=norm_layer, leaky_slope=leaky_slope,
                             dropout_p=dropout_flag*dropout_p, activation=True)]
            input_dim = down_dim_2
            # dropout for every other layer
            dropout_flag = not dropout_flag
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(down_dim_2, 1, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y