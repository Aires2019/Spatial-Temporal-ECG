from model.layers.base_layers import *
from model.layers.time_features import *
from model.layers.graph_learning import *
import torch.nn as nn


class STAR(nn.Module):
    def __init__(self, space_out_dim, time_out_dim, conv_kernel, hidden_dim, time_length, num_node,
                 num_windows, decay, pool_ratio, n_class):
        super(STAR, self).__init__()
        self.time_feature = MTFE(input_channels=1, hidden_dim=hidden_dim, output_dim=time_out_dim, single_view=True)
        self.nonlin_map = nn.Sequential(
            nn.Linear(time_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.positional_encoding = PositionalEncoding(hidden_dim, 0.1, max_len=5000)
        self.transformer = Transformer(dim=hidden_dim, heads=8, dropout=0.3)
        self.space_feature = GraphMPNN_block_2layers(hidden_dim, space_out_dim, num_node, time_length, decay,
                                                     pool_ratio, 2, conv_kernel, type='lead')
        self.kan_fc = KAN([space_out_dim * num_windows * num_node, 128, 128, n_class],
                          base_activation=torch.nn.SiLU)

    def forward(self, X, return_features=False):
        bs, tlen, num_node, dimension = X.size()
        # Graph Generation
        A_input = torch.reshape(X, [bs * tlen * num_node, dimension, 1])
        A_input = A_input.transpose(1, 2)
        A_input_ = self.time_feature(A_input)
        A_input_ = torch.reshape(A_input_, [bs * tlen * num_node, -1])
        A_input_ = self.nonlin_map(A_input_)
        A_input_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])

        X_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])
        X_ = torch.transpose(X_, 1, 2)
        X_ = torch.reshape(X_, [bs * num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = self.transformer(X_)
        transformer_feature = X_

        X_ = torch.reshape(X_, [bs, num_node, tlen, -1])
        X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_

        # space feature extract
        MPNN_output = self.space_feature(A_input_)

        space_features = MPNN_output
        features = torch.reshape(MPNN_output, [bs, -1])
        # print("features", features.size())

        features = self.kan_fc(features)
        if return_features:
            avgpool = nn.AdaptiveAvgPool1d(64)
            return space_features
        else:
            return features
