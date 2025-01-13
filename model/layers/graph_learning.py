import torch.nn as nn
import torch
from model.layers.space_feature import *
from model.layers.base_layers import *

'''
Here the code places a dynamic graph learning network
'''


class GraphMPNN_block_1layer(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, decay, pool_ratio, merge_size, kernel_size,
                 type='normal'):
        super(GraphMPNN_block_1layer, self).__init__()
        num_layers = 1
        num_nodes = num_sensors * time_length
        left_num_nodes = []
        left_node = time_length
        for layer in range(num_layers + 1):
            if left_node > 0:
                if type == 'normal':  # 一般情况下用池化率
                    left_node = round(num_nodes * (1 - (pool_ratio * layer)))
                    left_num_nodes.append(left_node)
                else:
                    left_num_nodes.append(left_node)
                    left_node = math.ceil(left_node / merge_size)
            else:
                left_num_nodes.append(1)
        paddings = [(k - 1) // 2 for k in kernel_size]
        # self.graph_construction = Graph_Construction_Similarity(num_nodes)
        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        # self.graph_construction = Multi_shallow_embedding(num_nodes, num_nodes)
        # self.graph_construction = Manual_Graph_Construction()

        self.graph_conv_gcn = nn.ModuleList(
            [GCN(input_dim, output_dim, k=1)]
        )
        self.graph_conv_gin = nn.ModuleList(
            [GIN(input_dim, output_dim, k=1)]
        )
        self.graph_conv_gat = nn.ModuleList(
            [GAT(input_dim, output_dim, dropout=0.5, alpha=0.2)]
        )
        self.graph_conv_sage = nn.ModuleList(
            [SAGEConv(input_dim, output_dim)]
        )
        self.graph_timepool = nn.ModuleList(
            [Dense_TimeDiffPool1d(left_num_nodes[0], left_num_nodes[1], kernel_size[0], paddings[0])]
        )
        self.graph_structpool = nn.ModuleList(
            [StructPool(left_num_nodes[1])]
        )
        self.graph_leadpool = nn.ModuleList(
            [LeadSpecificPatchPool_new(output_dim, left_num_nodes[0], left_num_nodes[1])]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(left_num_nodes[1])]
        )
        self.bns_lead = nn.ModuleList(
            [nn.BatchNorm1d(left_num_nodes[1] * num_sensors)]
        )
        self.relu = nn.ReLU()
        # self.pre_relation = Mask_Matrix_patch(num_sensors, time_length, decay)

    def forward(self, input):
        bs, tlen, num_sensors, feature_dim = input.size()
        x = torch.reshape(input, [bs, tlen * num_sensors, feature_dim])
        adj = self.graph_construction(x)
        # adj = adj * self.pre_relation
        for gconv, pool, bn in zip(self.graph_conv_gin, self.graph_leadpool, self.bns_lead):
            x = gconv(x, adj)
            x, adj = pool(x, adj)
            x = self.relu(bn(x))
            x = F.dropout(x, p=0.3)
        return x


class GraphMPNN_block_2layers(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, decay, pool_ratio, merge_size, kernel_size,
                 type='normal'):
        super(GraphMPNN_block_2layers, self).__init__()
        num_layers = 2
        num_nodes = num_sensors * time_length
        left_num_nodes = []
        left_node = time_length
        for layer in range(num_layers + 1):
            if left_node > 0:
                if type == 'normal':
                    left_node = round(num_nodes * (1 - (pool_ratio * layer)))
                    left_num_nodes.append(left_node)
                else:
                    left_num_nodes.append(left_node)
                    left_node = math.ceil(left_node / merge_size)
            else:
                left_num_nodes.append(1)
        paddings = [(k - 1) // 2 for k in kernel_size]
        # self.graph_construction = Graph_Construction_Similarity(num_nodes)
        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        # self.graph_construction = Multi_shallow_embedding(num_nodes, 16)
        # self.graph_construction = Manual_Graph_Construction()

        self.graph_conv_gcn = nn.ModuleList(
            [GCN(input_dim, output_dim, k=1)] +
            [GCN(output_dim, 2 * output_dim, k=1)]
        )
        self.graph_conv_gin = nn.ModuleList(
            [GIN(input_dim, output_dim, k=1)] +
            [GIN(output_dim, 2 * output_dim, k=1)]
        )
        # self.graph_conv_gat = nn.ModuleList(
        #     [GAT(input_dim, output_dim, dropout=0.5, alpha=0.2)] +
        #     [GAT(output_dim, 2 * output_dim, dropout=0.5, alpha=0.2)]
        # )
        # self.graph_conv_sage = nn.ModuleList(
        #     [SAGEConv(input_dim, output_dim)] +
        #     [SAGEConv(output_dim, 2 * output_dim)]
        # )
        self.graph_timepool = nn.ModuleList(
            [Dense_TimeDiffPool1d(left_num_nodes[0], left_num_nodes[1], kernel_size[0], paddings[0])] +
            [Dense_TimeDiffPool1d(left_num_nodes[1], left_num_nodes[2], kernel_size[1], paddings[1])]
        )
        # self.graph_structpool = nn.ModuleList(
        #     [StructPool(left_num_nodes[1])] +
        #     [StructPool(left_num_nodes[2])]
        # )
        self.graph_leadpool = nn.ModuleList(
            [LeadSpecificPatchPool_new(output_dim, left_num_nodes[0], left_num_nodes[1])] +
            [LeadSpecificPatchPool_new(2 * output_dim, left_num_nodes[1], left_num_nodes[2])]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(left_num_nodes[1])] +
            [nn.BatchNorm1d(left_num_nodes[2])]
        )
        self.bns_lead = nn.ModuleList(
            [nn.BatchNorm1d(left_num_nodes[1] * num_sensors)] +
            [nn.BatchNorm1d(left_num_nodes[2] * num_sensors)]
        )
        # self.relu = Mish()
        self.relu = nn.ReLU()
        self.pre_relation = Mask_Matrix_patch(num_sensors, time_length, decay)

    def forward(self, input):
        bs, tlen, num_sensors, feature_dim = input.size()
        x = torch.reshape(input, [bs, tlen * num_sensors, feature_dim])

        adj = self.graph_construction(x)
        original_adj = adj
        adj = adj * self.pre_relation
        # print("adj",self.pre_relation.shape)

        for gconv, pool, bn in zip(self.graph_conv_gin, self.graph_leadpool, self.bns_lead):
            x = gconv(x, adj)
            # print("after gcn", x.shape)
            x, adj = pool(x, adj)
            # print("after pool", x.shape)
            x = self.relu(bn(x))
            x = F.dropout(x, p=0.3)
        return x


class GraphMPNN_block_3layers(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, decay, pool_ratio, merge_size, kernel_size,
                 type='normal'):
        super(GraphMPNN_block_3layers, self).__init__()
        num_layers = 3
        num_nodes = num_sensors * time_length
        left_num_nodes = []
        left_node = time_length
        for layer in range(num_layers + 1):
            if left_node > 0:
                if type == 'normal':
                    left_node = round(num_nodes * (1 - (pool_ratio * layer)))
                    left_num_nodes.append(left_node)
                else:
                    left_num_nodes.append(left_node)
                    left_node = math.ceil(left_node / merge_size)
            else:
                left_num_nodes.append(1)
        paddings = [(k - 1) // 2 for k in kernel_size]

        # self.graph_construction = Graph_Construction_Similarity(num_nodes)
        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        # self.graph_construction = Multi_shallow_embedding(num_nodes, num_nodes, 16)
        # self.graph_construction = Manual_Graph_Construction()

        self.graph_conv_gcn = nn.ModuleList(
            [GCN(input_dim, output_dim, k=1)] +
            [GCN(output_dim, 2 * output_dim, k=1)] +
            [GCN(2 * output_dim, output_dim, k=1)]
        )
        self.graph_conv_gin = nn.ModuleList(
            [GIN(input_dim, output_dim, k=1)] +
            [GIN(output_dim, 2 * output_dim, k=1)] +
            [GIN(2 * output_dim, output_dim, k=1)]
        )

        self.graph_timepool = nn.ModuleList(
            [Dense_TimeDiffPool1d(left_num_nodes[0], left_num_nodes[1], kernel_size[0], paddings[0])] +
            [Dense_TimeDiffPool1d(left_num_nodes[1], left_num_nodes[2], kernel_size[1], paddings[1])] +
            [Dense_TimeDiffPool1d(left_num_nodes[2], left_num_nodes[3], kernel_size[2], paddings[2])]
        )
        self.graph_structpool = nn.ModuleList(
            [StructPool(left_num_nodes[1])] +
            [StructPool(left_num_nodes[2])] +
            [StructPool(left_num_nodes[3])]
        )
        self.graph_leadpool = nn.ModuleList(
            [LeadSpecificPatchPool(output_dim, left_num_nodes[0], left_num_nodes[1])] +
            [LeadSpecificPatchPool(2 * output_dim, left_num_nodes[1], left_num_nodes[2])] +
            [LeadSpecificPatchPool(output_dim, left_num_nodes[2], left_num_nodes[3])]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(left_num_nodes[1])] +
            [nn.BatchNorm1d(left_num_nodes[2])] +
            [nn.BatchNorm1d(left_num_nodes[3])]
        )
        self.bns_lead = nn.ModuleList(
            [nn.BatchNorm1d(left_num_nodes[1] * num_sensors)] +
            [nn.BatchNorm1d(left_num_nodes[2] * num_sensors)] +
            [nn.BatchNorm1d(left_num_nodes[3] * num_sensors)]
        )
        self.relu = Mish()
        self.pre_relation = Mask_Matrix_patch(num_sensors, time_length, decay)

    def forward(self, input):
        bs, tlen, num_sensors, feature_dim = input.size()
        x = torch.reshape(input, [bs, tlen * num_sensors, feature_dim])

        adj = self.graph_construction(x)

        adj = adj * self.pre_relation
        for gconv, pool, bn in zip(self.graph_conv_gin, self.graph_leadpool, self.bns_lead):
            x = gconv(x, adj)
            x, adj = pool(x, adj)
            x = self.relu(bn(x))
            x = F.dropout(x, p=0.3)
        return x

