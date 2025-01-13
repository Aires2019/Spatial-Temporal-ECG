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

# 此处代码放置图学习的模块

'''
建图的方法
'''


# 动态图构建 利用节点之间的相似性==========================================================================================
class Graph_Construction_Similarity(nn.Module):
    def __init__(self, num_node):
        super(Graph_Construction_Similarity, self).__init__()
        self.W_adj = nn.Linear(num_node, num_node, bias=False)

    def forward(self, X):
        # 计算相似性矩阵C
        dist = torch.cdist(X, X)  # 计算每对节点之间的欧氏距离
        sim = torch.exp(-dist) / torch.sum(torch.exp(-dist), dim=-1, keepdim=True)  # batch_size*num_node*num_node
        # 生成邻接矩阵A
        A = torch.sigmoid(self.W_adj(sim))  # 激活函数可以是sigmoid或softmax

        # 稀疏化和归一化
        threshold = 0.3
        A = torch.where(A > threshold, A, torch.tensor(0.0, device=A.device))
        A = A / A.sum(dim=-1, keepdim=True)  # 行归一化

        return A


# FC图构建 直接利用节点特征相乘进行初始化=====================================================================================
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


# 不用到特征矩阵 只进行随机初始化建图=========================================================================================
class Multi_shallow_embedding(nn.Module):
    def __init__(self, num_nodes, k_neighs):
        super().__init__()

        self.num_nodes = num_nodes
        self.k = k_neighs
        # num_graphs will be dynamic, so it's not initialized here

    def reset_parameters(self, emb_s, emb_t):
        init.xavier_uniform_(emb_s)
        init.xavier_uniform_(emb_t)

    def forward(self, x):
        # Extract batch size as num_graphs from x
        num_graphs = x.shape[0]  # assuming x has shape [batch_size, ...]
        device = x.device
        # Create emb_s and emb_t dynamically based on the current batch size
        emb_s = Parameter(torch.Tensor(num_graphs, self.num_nodes, 1)).to(device)
        emb_t = Parameter(torch.Tensor(num_graphs, 1, self.num_nodes)).to(device)

        # Initialize the parameters
        self.reset_parameters(emb_s, emb_t)

        # adj: [G, N, N]
        adj = torch.matmul(emb_s, emb_t).to(device)

        # Remove self-loops
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')

        # top-k-edge adj
        adj_flat = adj.reshape(num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)

        idx = torch.tensor([i // self.k for i in range(indices.size(0))], device=device)

        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.
        adj = adj_flat.reshape_as(adj)

        return adj


# 手动图构建======================================================================================================
class Manual_Graph_Construction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_features):
        bs, N, dimen = node_features.size()
        # 初始化12x12的邻接矩阵，所有元素都为0
        # adj_matrix = np.zeros((N, N), dtype=np.float)
        adj_matrix = torch.full((N, N), 0.0, dtype=torch.float32).cuda(args.gpu)
        # 节点1到节点6全连接
        for i in range(1, 7):
            for j in range(1, 7):
                adj_matrix[i - 1][j - 1] = 1.0

        # 节点7到节点12全连接
        for i in range(7, 13):
            for j in range(7, 13):
                adj_matrix[i - 1][j - 1] = 1.0

        # 节点1、节点6、节点10、节点11全连接
        adj_matrix[0][5] = 1.0
        adj_matrix[5][0] = 1.0
        adj_matrix[0][9] = 1.0
        adj_matrix[0][10] = 1.0
        adj_matrix[5][9] = 1.0
        adj_matrix[5][10] = 1.0
        adj_matrix[9][0] = 1.0
        adj_matrix[9][5] = 1.0
        adj_matrix[10][0] = 1.0
        adj_matrix[10][5] = 1.0

        # 适应batch size
        adj_batch = adj_matrix.repeat(bs, 1, 1)
        adj_batch = adj_batch.cuda(args.gpu)
        return adj_batch


"""
池化的方法
"""


# 时间图池化方案==========================================================================================
class Dense_TimeDiffPool1d(nn.Module):
    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()

        # TODO: add Normalization
        self.time_conv = nn.Conv1d(pre_nodes, pooled_nodes, kern_size, padding=padding)

        self.re_param = Parameter(Tensor(kern_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')

    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, N, F]
            adj (Tensor): [G, N, N]
        """

        out = self.time_conv(x)

        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))
        # print("out_adj", out_adj.size())  # [16,48,48] -> [16,36,36] -> [16,24,24]

        return out, out_adj


# StructPool 结构图池化方法=============================================================================================
class StructPool(nn.Module):

    def __init__(self, k, latent_dim=[48, 48]):
        super(StructPool, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim.append(k)
        self.k = k  # 聚类数量
        self.number_iterations = 5
        self.l_hop = 15
        self.dense_crf = True
        self.softmax = nn.Softmax(dim=None)
        self.w_filter = Variable(torch.eye(k)).float().cuda(args.gpu)
        self.w_compat = Variable(-1 * torch.eye(k)).float().cuda(args.gpu)
        self.avgpool = nn.AdaptiveAvgPool1d(k)

    def calculate_U(self, X, A):
        b, n, _ = A.size()
        U = torch.bmm(A, X)
        U = self.avgpool(U)
        # 行归一化
        normalized_linear = U / U.sum(dim=-1, keepdim=True)  # 行归一化
        U = torch.tanh(normalized_linear)
        return U

    def forward(self, X, A):
        A = A.float()
        n2n_sp = A
        U = self.calculate_U(X, A)
        q_values = U
        '''  "Perform crf pooling" '''
        for i in range(self.number_iterations):
            '''  Step one, softmax as initialize, unary potentials U across all the labels at each node '''
            softmax_out = F.softmax(q_values, dim=-1)  # [b,n,k]
            ''' Use vector similarity to replace kernels '''
            matrix_W = torch.matmul(X, torch.transpose(X, -2, -1)).float()  # [b,n,n]
            Diag = torch.eye(matrix_W.size()[-2], matrix_W.size()[-1])
            Diag = Diag.view(Diag.size()[-2], -1).float().cuda(args.gpu)

            W = matrix_W - matrix_W * Diag  # [b,n,n]

            if not self.dense_crf:
                A_l = self.get_l_hops(A, self.l_hop)
                W = W * A_l

            normalized_m = torch.sum(W, dim=-1, keepdim=True)  # [b,n,1]
            out = torch.matmul(W, softmax_out)  # [b,n,k]
            out_norm = torch.div(out, normalized_m)  # [b,n,k]
            '''' weighting filter outputs'''
            out_norm = torch.matmul(out_norm, self.w_filter)  # [b,n,k]
            ''' Next, Compatibility Transform '''
            out_norm = torch.matmul(out_norm, self.w_compat)  # [b,n,k]
            q_values = U - out_norm  # [b,n,k]

        L = F.softmax(q_values, dim=-1)  # [b,n,k]
        L_onehot = L
        L_onehot_T = torch.transpose(L_onehot, -2, -1)  # [b,k,n]
        X_out = torch.matmul(L_onehot_T, X)  # [b,k,d]

        A_out0 = torch.matmul(L_onehot_T, A)
        A_out = torch.matmul(A_out0, L_onehot)
        return X_out, A_out


# 按导联特定的按照patch池化===============================================================================================
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
        按导联进行独立池化，每个导联的patch使用卷积处理，不跨导联池化
        Args:
            x: 输入特征矩阵，形状为 [batch_size, num_nodes, features]
            adj: 输入邻接矩阵，形状为 [batch_size, num_nodes, num_nodes]
        Returns:
            new_x: 池化后的特征矩阵
            new_adj: 池化后的邻接矩阵
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
        按导联进行独立池化，每个导联的patch使用卷积处理，不跨导联池化
        Args:
            x: 输入特征矩阵，形状为 [batch_size, num_nodes, features]
            adj: 输入邻接矩阵，形状为 [batch_size, num_nodes, num_nodes]
        Returns:
            pooled_x: 池化后的特征矩阵
            pooled_adj: 池化后的邻接矩阵
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


# 变分图池化方案==========================================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class VariationalGraphPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters):
        super(VariationalGraphPooling, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, num_clusters * input_dim)
        self.num_clusters = num_clusters

    def forward(self, x):
        # x: [batch_size, num_nodes, input_dim]
        batch_size, num_nodes, input_dim = x.size()

        # Encode
        x_encoded = self.encoder(x)  # [batch_size, num_nodes, hidden_dim]

        # Decode to get cluster centers
        cluster_centers = self.decoder(x_encoded).view(batch_size, num_nodes, self.num_clusters, input_dim)
        cluster_centers = cluster_centers.mean(dim=1)  # [batch_size, num_clusters, input_dim]

        # Compute assignment matrix using cosine similarity
        x_norm = x / x.norm(dim=-1, keepdim=True)  # Normalize
        cluster_centers_norm = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)  # Normalize
        assignment_matrix = torch.matmul(x_norm, cluster_centers_norm.transpose(-1,
                                                                                -2))  # [batch_size, num_nodes, num_clusters]
        assignment_matrix = F.softmax(assignment_matrix, dim=-1)

        # Aggregate features
        pooled_features = torch.matmul(assignment_matrix, x)  # [batch_size, num_clusters, input_dim]

        return pooled_features, assignment_matrix, cluster_centers


'''
图特征提取的网络方法
'''


# FC图卷积
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


# 图同构网络GIN
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
