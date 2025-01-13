import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from torch import Tensor
from torch.nn import Parameter, init

from args import args
from model.layers.kan_layers import KANLinear, FastKANConv1DLayer, KAGNConv1DLayer, L1

args = args()


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (tr.tanh(F.softplus(x)))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = tr.zeros(max_len, d_model).cuda(args.gpu)
        position = tr.arange(0, max_len).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, d_model, 2) *
                          -(math.log(100.0) / d_model))
        pe[:, 0::2] = tr.sin(position * div_term)
        pe[:, 1::2] = tr.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x


def Mask_Matrix_patch(num_node, time_length, decay_rate):
    epsilon = 0
    adj_matrix = tr.full((num_node * time_length, num_node * time_length), epsilon).cuda(args.gpu)

    num_leads = num_node
    patches_per_lead = time_length
    decay_factor = decay_rate

    for lead in range(num_leads):
        start_idx = lead * patches_per_lead
        end_idx = start_idx + patches_per_lead

        for i in range(patches_per_lead):
            for j in range(i, patches_per_lead):
                distance = abs(i - j)
                adj_matrix[start_idx + i, start_idx + j] = decay_factor ** distance
                adj_matrix[start_idx + j, start_idx + i] = decay_factor ** distance  # 对称矩阵

    for lead_group in [(0, 6), (6, 12)]:
        for lead1 in range(lead_group[0], lead_group[1]):
            for lead2 in range(lead1 + 1, lead_group[1]):
                start1, end1 = lead1 * patches_per_lead, (lead1 + 1) * patches_per_lead
                start2, end2 = lead2 * patches_per_lead, (lead2 + 1) * patches_per_lead
                adj_matrix[start1:end1, start2:end2] = 1 * decay_rate
                adj_matrix[start2:end2, start1:end1] = 1 * decay_rate

    special_leads = [0, 5, 9, 10]
    for i, lead1 in enumerate(special_leads):
        for lead2 in special_leads[i + 1:]:
            start1, end1 = lead1 * patches_per_lead, (lead1 + 1) * patches_per_lead
            start2, end2 = lead2 * patches_per_lead, (lead2 + 1) * patches_per_lead
            adj_matrix[start1:end1, start2:end2] = 1 * decay_rate
            adj_matrix[start2:end2, start1:end1] = 1 * decay_rate
    return adj_matrix


class KAN(tr.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=tr.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = tr.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(tr.nn.BatchNorm1d(in_features))
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                ))

            # self.layers.append(tr.nn.BatchNorm1d(out_features))

    def forward(self, x: tr.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# Transformer+KAN --------------------------------------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super(Transformer, self).__init__()
        self.layers = Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout)))

    def forward(self, x, mask=None):
        x = self.layers(x, mask=mask)  # go to attention
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = KAN_GPT(width=[dim, 3 * dim])

        self.nn1 = KAN_GPT(width=[dim, dim])

        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = tr.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -tr.finfo(dots.dtype).max

        if mask is not None:
            mask = nn.functional.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = tr.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class KAN_GPT(tr.nn.Module):
    def __init__(
            self,
            width,
            grid=3,
            k=3,
            noise_scale=0.1,
            noise_scale_base=1.0,
            scale_spline=1.0,
            base_fun=tr.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            bias_trainable=True,
    ):
        super(KAN_GPT, self).__init__()
        self.grid_size = grid
        self.spline_order = k
        self.bias_trainable = bias_trainable

        self.layers = tr.nn.ModuleList()
        for in_features, out_features in zip(width, width[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid,
                    spline_order=grid,
                    scale_noise=noise_scale,
                    scale_base=noise_scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_fun,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )

            )
            self.layers.append(tr.nn.BatchNorm1d(out_features))

    def forward(self, x: tr.Tensor, update_grid=False):
        B, C, T = x.shape

        x = x.view(-1, T)

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)

        U = x.shape[1]

        x = x.view(B, C, U)

        return x

    def regularization_loss(
            self, regularize_activation=1.0, regularize_entropy=1.0
    ):
        return sum(
            layer.regularization_loss(
                regularize_activation, regularize_entropy
            )
            for layer in self.layers
        )
