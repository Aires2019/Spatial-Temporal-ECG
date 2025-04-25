import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.autograd import Variable
from FC_STGNN.args import args

args = args()
# from pytorch_util import weights_init, gnn_spmm
EPS = 1e-15

'''
This is for Graph Learning
'''


class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = torch.transpose(node_features, 1, 2)

        Adj = torch.bmm(node_features, node_features_1)

        eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda(args.gpu)
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj


"""
Graph Pooling
"""


class AdaptiveWeight(nn.Module):
    def __init__(self, plances=32):
        super(AdaptiveWeight, self).__init__()

        self.fc = nn.Linear(plances, 1)
        # self.bn = nn.BatchNorm1d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        # out = self.bn(out)
        out = self.sig(out)

        return out


class LeadSpecificPatchPool(nn.Module):
    def __init__(self, weight_dim, num_patches_per_lead, target_patches_per_lead, kernel_size=3, padding=1):
        super().__init__()
        self.per_lead = num_patches_per_lead
        self.after_lead = target_patches_per_lead
        self.conv1d = nn.Conv1d(in_channels=num_patches_per_lead,
                                out_channels=target_patches_per_lead,
                                kernel_size=kernel_size,
                                padding=padding)
        # 定义adj的池化
        # 使用卷积层进行池化
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=padding, bias=False)
        # 设置卷积权重为均值池化
        self.conv.weight = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size),
                                        requires_grad=False)
        # 权重
        self.fuse_weight = AdaptiveWeight(weight_dim)

    def forward(self, x, adj):
        """
        Independent pooling is performed by lead, and the patch of each lead is processed by convolution, without cross-lead pooling
        Args:
            x: Input feature matrix, shape is [batch_size, num_nodes, features]
            adj: Input adjacency matrix, shape is [batch_size, num_nodes, num_nodes]
        Returns:
            new_x: Pooled feature matrix
            new_adj: Pooled adjacency matrix
        """
        batch_size, num_nodes, features = x.shape
        adj = adj.unsqueeze(1)
        batch_size, channels, height, width = adj.shape
        num_leads = num_nodes // self.per_lead  # 每个导联的patch数 60/5=12
        new_x = []
        new_x_weight = []

        # 分导联进行卷积池化
        for lead in range(num_leads):
            start_idx = lead * self.per_lead

            x_lead = x[:, start_idx:start_idx + self.per_lead, :]  # 提取当前导联的patch
            # 卷积操作 对特征矩阵进行卷积
            pooled_x_lead = self.conv1d(x_lead)

            # 加权重
            weight = self.fuse_weight(pooled_x_lead)
            new_x_weight.append(weight)
            new_x.append(pooled_x_lead)

        pooled_adj = torch.zeros(batch_size, 1, height // self.per_lead * self.after_lead,
                                 width // self.per_lead * self.after_lead, device=adj.device)

        # 遍历导联（按每 5*5 块的方式处理）
        for i in range(0, height, self.per_lead):
            for j in range(0, width, self.per_lead):
                # 对每个 5x5 的子矩阵进行卷积池化
                sub_adj = adj[:, :, i:i + self.per_lead, j:j + self.per_lead]
                pooled_block = self.conv(sub_adj)

                # 将池化后的结果放到对应位置
                pooled_adj[:, :, i // self.per_lead * self.after_lead:(i // self.per_lead + 1) * self.after_lead,
                j // self.per_lead * self.after_lead:(j // self.per_lead + 1) * self.after_lead] = pooled_block

        x_output = []
        for lead in range(num_leads):
            xx = new_x[lead] * new_x_weight[lead]
            x_output.append(xx)
        x_output = torch.cat(x_output, dim=1)

        return x_output, pooled_adj.squeeze(1)


# 导联池化新方案==========================================================================================
class LeadSpecificPatchPool_new(nn.Module):
    def __init__(self, weight_dim, num_patches_per_lead, target_patches_per_lead, kernel_size=3, padding=1):
        super().__init__()
        self.per_lead = num_patches_per_lead
        self.after_lead = target_patches_per_lead
        self.conv2d = nn.Conv2d(in_channels=num_patches_per_lead,
                                out_channels=target_patches_per_lead,
                                kernel_size=(1, kernel_size),
                                padding=(0, padding))
        self.re_param = Parameter(Tensor(kernel_size, 1))

    def forward(self, x, adj):
        """
        Independent pooling is performed by lead, and each lead patch is processed by convolution, without cross-lead pooling
        Args:
            x: Input feature matrix, shape is [batch_size, num_nodes, features]
            adj: Input adjacency matrix, shape is [batch_size, num_nodes, num_nodes]
        Returns:
            pooled_x: Pooled feature matrix
            pooled_adj: Pooled adjacency matrix
        """

        batch_size, num_nodes, features = x.shape
        x = x.reshape(batch_size, self.per_lead, -1, features)
        x = self.conv2d(x)
        s = torch.matmul(self.conv2d.weight, self.re_param).view(-1, self.per_lead)
        batch_size, height, width = adj.shape
        pooled_adj = torch.zeros(batch_size, height // self.per_lead * self.after_lead,
                                 width // self.per_lead * self.after_lead, device=adj.device)
        for i in range(0, height, self.per_lead):
            for j in range(0, width, self.per_lead):
                # 对每个 5x5 的子矩阵进行卷积池化
                sub_adj = adj[:, i:i + self.per_lead, j:j + self.per_lead]
                pooled_block = torch.matmul(torch.matmul(s, sub_adj), s.transpose(0, 1))
                pooled_adj[:, i // self.per_lead * self.after_lead:(i // self.per_lead + 1) * self.after_lead,
                j // self.per_lead * self.after_lead:(j // self.per_lead + 1) * self.after_lead] = pooled_block
        pooled_x = x.reshape(batch_size, -1, features)
        return pooled_x, pooled_adj

'''
Network Methods for Graph Feature Extraction
'''


class GCN(nn.Module):
    def __init__(self, input_dimension, output_dinmension, k):
        # In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        # k=1 means the traditional GCN
        super(GCN, self).__init__()
        self.way_multi_field = 'sum'  # two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, output_dinmension))
        self.theta = nn.ModuleList(theta)
        self.bn1 = nn.BatchNorm1d(output_dinmension)

    def forward(self, X, A):
        # size of X is (bs, N, A)
        # size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_, A)
            out_k = self.theta[kk](torch.bmm(A_, X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = torch.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        GCN_output_ = torch.transpose(GCN_output_, -1, -2)
        GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = torch.transpose(GCN_output_, -1, -2)

        return F.leaky_relu(GCN_output_)


class GIN(nn.Module):
    def __init__(self, input_dimension, output_dimension, k):
        # k in GIN is typically the number of layers
        super(GIN, self).__init__()
        self.k = k
        self.eps = nn.Parameter(torch.zeros(k))
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dimension, output_dimension),
            nn.ReLU(),
            nn.Linear(output_dimension, output_dimension)
        ) for _ in range(k)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(output_dimension) for _ in range(k)])

    def norm(self, adj, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj

    def forward(self, X, A):
        # size of X is (bs, N, D)
        # size of A is (bs, N, N)
        A = self.norm(A, add_loop=False)
        h = X
        for i in range(self.k):
            h = self.mlp[i]((1 + self.eps[i]) * h + torch.bmm(A, h))
            h = torch.transpose(h, -1, -2)
            h = self.batch_norms[i](h)
            h = torch.transpose(h, -1, -2)
        return h


class GAT(nn.Module):
    """
    Batch-compatible Graph Attention Network (GAT) layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Learnable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [batch_size, N, in_features]
        adj: adjacency matrix [batch_size, N, N]
        """
        batch_size, N, _ = inp.size()

        # Apply linear transformation
        h = torch.matmul(inp, self.W)  # [batch_size, N, out_features]

        # Compute attention scores
        h_repeat_interleave = h.repeat(1, 1, N).view(batch_size, N * N, -1)
        h_repeat_tile = h.repeat(1, N, 1)
        a_input = torch.cat([h_repeat_interleave, h_repeat_tile], dim=-1).view(batch_size, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [batch_size, N, N]

        # Masking and softmax normalization
        zero_vec = -1e12 * torch.ones_like(e)  # Mask for non-connected nodes
        attention = torch.where(adj > 0, e, zero_vec)  # [batch_size, N, N]
        attention = F.softmax(attention, dim=-1)  # Normalize attention scores
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention weights
        h_prime = torch.bmm(attention, h)  # [batch_size, N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SAGEConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(SAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features)  # 将输入特征映射到输出特征
        self.out_proj = nn.Linear(2 * out_features, out_features)  # 拼接后的特征映射

    def forward(self, x, adj):
        """
        x: 输入特征，形状为 [batch_size, N, in_features]
        adj: 邻接矩阵，形状为 [batch_size, N, N]
        """
        batch_size, N, _ = x.size()

        # 计算线性变换后的支持项
        support = self.proj(x)  # [batch_size, N, out_features]

        # 计算邻接矩阵的归一化
        eps = 1e-8
        row_sums = adj.sum(dim=2, keepdim=True)  # [batch_size, N, 1]，每个节点的邻接行和
        row_sums = torch.max(row_sums, eps * torch.ones_like(row_sums))  # 防止除以零
        normalized_adj = adj / row_sums  # 归一化邻接矩阵

        # 计算邻居聚合
        # torch.einsum('bni,bnd->bnd', normalized_adj, support) 计算邻接矩阵与特征矩阵的加权和
        output = torch.bmm(normalized_adj, support)  # [batch_size, N, out_features]

        # 拼接 support 和 output
        cat_x = torch.cat((support, output), dim=-1)  # [batch_size, N, 2 * out_features]

        # 投影到最终输出
        z = self.out_proj(cat_x)  # [batch_size, N, out_features]

        # L2 正则化
        z_norm = z.norm(p=2, dim=-1, keepdim=True)  # [batch_size, N, 1]
        z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)  # 防止零除
        z = z / z_norm  # L2 正则化

        return z

# if __name__ == "__main__":
#     batch_size, num_nodes, channels, num_clusters = (16, 12, 32, 6)
#     x = torch.randn((batch_size, num_nodes, channels))
#     adj = torch.rand((batch_size, num_nodes, num_nodes))
#     u = torch.randn((batch_size, num_nodes, num_clusters))  # 潜在变量矩阵
#     mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool)
#
#     model = StructPool(num_clusters)
#     pool_x, pool_adj = model(x, adj)
#     print("pool_x", pool_x.shape)
#     print("pool_adj", pool_adj.shape)
