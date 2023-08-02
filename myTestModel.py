from models import temporalAttention
import argparse
import ast
import configparser
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.masking import FullMask, LengthMask
from utils import construct_adj_local, get_adjacency_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, graph_A):
        x = torch.einsum('ncvl,vw->ncwl', (x, graph_A))
        return x.contiguous()


class gcn_diff(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn_diff, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.projection = torch.nn.Conv2d(c_in,
                                          c_out,
                                          kernel_size=(1, 1),
                                          padding=(0, 0),
                                          stride=(1, 1),
                                          bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x: [B, D, N, T]
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.projection(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h  # [B, D, N, T]


class TemporalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super(TemporalEncoding, self).__init__()

        self.dayofweek_emb = nn.Embedding(7, d_model)
        self.timeofday_emb = nn.Embedding(24 * 12, d_model)

    def forward(self, x_mark):
        x_mark = x_mark.long()

        dayofweek_x = self.dayofweek_emb(x_mark[:, :, :, 0])
        timeofday_x = self.timeofday_emb(x_mark[:, :, :, 1])

        return dayofweek_x + timeofday_x


class MeterlogicalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super(MeterlogicalEncoding, self).__init__()

        self.dayofweek_emb = nn.Embedding(7, d_model)
        self.timeofday_emb = nn.Embedding(24 * 12, d_model)

    def forward(self, x_mark):
        x_mark = x_mark.long()

        dayofweek_x = self.dayofweek_emb(x_mark[:, :, :, 0])
        timeofday_x = self.timeofday_emb(x_mark[:, :, :, 1])

        return dayofweek_x + timeofday_x


class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation="GLU"):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {"GLU", "relu"}

        if self.activation == "GLU":
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        adj = self.adj
        # if mask is not None:
        #     adj = adj.to(mask.device) * mask

        x = torch.einsum("nm, mbc->nbc", adj.to(x.device), x)  # 3*N, B, Cin
        # Add batch dimmension
        # x = torch.einsum("bnm, mbc->nbc", adj.to(x.device), x)  # 3*N, B, Cin

        if self.activation == "GLU":
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim,
                                   dim=-1)  # 3*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == "relu":
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数为sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数为cos

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(-2)


class STSGCM(nn.Module):
    def __init__(self,
                 adj,
                 in_dim,
                 out_dims,
                 num_of_vertices,
                 activation="GLU"):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation,
            ))

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i - 1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))

    def forward(self, x, mask=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(h[self.num_of_vertices:2 * self.num_of_vertices],
                            dim=0) for h in need_concat
        ]

        # (N, B, Cout) #MAXPOLLING OPERATION
        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values

        del need_concat

        return out


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, feature_map=None, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        # [B, N, L, H, d]
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, None, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("bnshd,bnshm->bnhmd", K, values)

        # Compute the normalizer
        Z = 1 / (torch.einsum("bnlhd,bnhd->bnlh", Q, K.sum(dim=2)) + self.eps)

        # Finally compute and return the new values
        V = torch.einsum("bnlhd,bnhmd,bnlh->bnlhm", Q, KV, Z)

        return V.contiguous()


class STSGCL(nn.Module):
    def __init__(
            self,
            adj,
            history,
            num_of_vertices,
            in_dim,
            out_dims,
            strides=3,
            activation="GLU",
            temporal_emb=True,
            spatial_emb=True,
    ):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.STSGCMS = nn.ModuleList()

        # for i in range(self.history - self.strides + 1):
        #     self.STSGCMS.append(
        #         STSGCM(
        #             adj=self.adj,
        #             in_dim=self.in_dim,
        #             out_dims=self.out_dims,
        #             num_of_vertices=self.num_of_vertices,
        #             activation=self.activation
        #         )
        #     )
        for _ in range(self.history):
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(
                torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin
        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(
                torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin
        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)
        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-2, N, Cout
        """
        """
        self.spatial_embedding: [1 1 307 64]
        self.temporal_embedding.shape [1 12 1 64] 
        x.shape [32 12 307 64] # 32个batch，12个时间步，307个节点，64个输入维度
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        need_concat = []
        batch_size = x.shape[0]
        pad_first = x[:, 0, :, :].unsqueeze(1)
        pad_last = x[:, -1, :, :].unsqueeze(1)
        x = torch.cat((pad_first, x, pad_last), dim=1)
        for i in range(self.history):  # self.history - self.strides + 1
            t = x[:, i:i + self.strides, :, :]  # (B, 3, N, Cin)
            # t_graphs = adj_list[:, i, :, :]  # (B, 3, N, N)

            t = torch.reshape(t,
                              shape=[
                                  batch_size,
                                  self.strides * self.num_of_vertices,
                                  self.in_dim
                              ])
            # (B, 3*N, Cin)

            # (3*N, B, Cin) -> (N, B, Cout)
            t = self.STSGCMS[i](t.permute(1, 0, 2), mask)

            # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)

            need_concat.append(t)

        # modified (B, T-2, N, Cout) -> (B, T, N, Cout)
        out = torch.cat(need_concat, dim=1)
        del need_concat, batch_size
        return out


class MULTISTSGCL(nn.Module):
    def __init__(
            self,
            adj_list,
            history,
            num_of_vertices,
            in_dim,
            out_dims,
            strides=3,
            activation="GLU",
            temporal_emb=True,
            spatial_emb=True,
    ):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(MULTISTSGCL, self).__init__()
        self.adjList = adj_list
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.spatial_STSGCMS = nn.ModuleList()
        self.pois_STSGCMS = nn.ModuleList()
        self.corr_STSGCMS = nn.ModuleList()
        self.LinearList1 = nn.ModuleList()
        # for i in range(self.history - self.strides + 1):
        #     self.STSGCMS.append(
        #         STSGCM(
        #             adj=self.adj,
        #             in_dim=self.in_dim,
        #             out_dims=self.out_dims,
        #             num_of_vertices=self.num_of_vertices,
        #             activation=self.activation
        #         )
        #     )

        for _ in range(3):
            self.LinearList1.append(
                nn.Linear(self.out_dims[0], self.out_dims[0]))
        self.OutLinear = nn.Linear(self.out_dims[0] * 3, self.out_dims[0])
        for _ in range(self.history):
            self.spatial_STSGCMS.append(
                STSGCM(
                    adj=self.adjList[0],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))

        for _ in range(self.history):
            self.pois_STSGCMS.append(
                STSGCM(
                    adj=self.adjList[1],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))

        for _ in range(self.history):
            self.corr_STSGCMS.append(
                STSGCM(
                    adj=self.adjList[2],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(
                torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin
        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(
                torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin
        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)
        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-2, N, Cout
        """
        """
        self.spatial_embedding: [1 1 307 64]
        self.temporal_embedding.shape [1 12 1 64] 
        x.shape [32 12 307 64] # 32个batch，12个时间步，307个节点，64个输入维度
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        need_concat = []
        batch_size = x.shape[0]
        pad_first = x[:, 0, :, :].unsqueeze(1)
        pad_last = x[:, -1, :, :].unsqueeze(1)
        x = torch.cat((pad_first, x, pad_last), dim=1)
        for i in range(self.history):  # self.history - self.strides + 1
            t = x[:, i:i + self.strides, :, :]  # (B, 3, N, Cin)
            # t_graphs = adj_list[:, i, :, :]  # (B, 3, N, N)

            t1 = torch.reshape(t,
                               shape=[
                                   batch_size,
                                   self.strides * self.num_of_vertices,
                                   self.in_dim
                               ])
            # (B, 3*N, Cin)
            t2 = t1.clone()
            t3 = t1.clone()

            # (3*N, B, Cin) -> (N, B, Cout)
            t1 = self.spatial_STSGCMS[i](t1.permute(1, 0, 2), mask)
            t2 = self.pois_STSGCMS[i](t2.permute(1, 0, 2), mask)
            t3 = self.corr_STSGCMS[i](t3.permute(1, 0, 2), mask)
            # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            t1 = F.relu(self.LinearList1[0](t1))
            t2 = F.relu(self.LinearList1[1](t2))
            t3 = F.relu(self.LinearList1[2](t3))

            t = torch.cat((t1, t2, t3), dim=-1)
            t = self.OutLinear(t)
            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)

            need_concat.append(t)

        # modified (B, T-2, N, Cout) -> (B, T, N, Cout)
        out = torch.cat(need_concat, dim=1)
        del need_concat, batch_size
        return out


class STSGCLdiffAttentionLayer(nn.Module):
    def __init__(
            self,
            attention,
            d_model,
            n_heads,
            adj_list,
            num_of_vertices,
            d_keys=None,
            d_values=None,
            support_len=1,
            order=2,
            dropout=0.0,
            use_mask=True,
            history=12,
            stride=3,
    ):
        super(STSGCLdiffAttentionLayer, self).__init__()
        self.use_mask = use_mask
        # Fill d_keys and d_values
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.adjList = adj_list
        self.num_of_vertices = num_of_vertices
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.activation = 'GLU'
        # self.gcn1 = gcn_operation(
        #     adj=self.adjList[0],
        #     in_dim=self.d_model,
        #     out_dim=d_model,
        #     num_vertices=self.num_of_vertices,
        #     activation=self.activation,
        # )
        # self.gcn2 = gcn_operation(
        #     adj=self.adjList[1],
        #     in_dim=self.d_model,
        #     out_dim=self.d_model,
        #     num_vertices=self.num_of_vertices,
        #     activation=self.activation,
        # )
        # self.gcn3 = gcn_operation(
        #     adj=self.adjList[2],
        #     in_dim=self.d_model,
        #     out_dim=self.d_model,
        #     num_vertices=self.num_of_vertices,
        #     activation=self.activation,
        # )

        self.gcn1 = gcn_diff(d_model,
                             d_model,
                             dropout=dropout,
                             support_len=support_len,
                             order=order)
        self.gcn2 = gcn_diff(d_model,
                             d_model,
                             dropout=dropout,
                             support_len=support_len,
                             order=order)
        self.sts1 = MULTISTSGCL(
            self.adjList,
            history=history,
            strides=stride,
            in_dim=d_model,
            out_dims=[d_model] * 3,
            num_of_vertices=self.num_of_vertices,
        )

        self.sts2 = MULTISTSGCL(
            self.adjList,
            history=history,
            strides=stride,
            in_dim=d_model,
            out_dims=[d_model] * 3,
            num_of_vertices=self.num_of_vertices,
        )
        self.adj = adj_list[0]
        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

    def forward(self,
                queries,
                keys,
                values,
                attn_mask,
                query_lengths,
                key_lengths,
                support=None):
        # Extract the dimensions into local variables
        support = self.adjList
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads
        # Q,K,V = B,T,N,D
        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)

        # gcn_queries = self.sts1(queries, mask=self.mask)
        # queries = self.query_projection(gcn_queries).view(B, L, N1, H, -1)

        # -> [B, T-2, N, Cout]
        gcn_keys = self.sts1(keys, mask=self.mask)
        # gcn_keys = self.gcn1(keys.transpose(-1, 1), support)  # [B, D, N, T]
        # keys = self.key_projection(
        #     gcn_keys.transpose(-1, 1)).view(B, S, N2, H, -1)  # [B, S, N, H, d]
        keys = self.key_projection(gcn_keys).view(B, S, N2, H, -1)
        # -> [B, T-2, N, Cout]
        gcn_values = self.sts2(values, mask=self.mask)
        # gcn_values = self.gcn2(values.transpose(-1, 1),
        #    support)  # [B, D, N, T]
        # values = self.value_projection(
        #     gcn_values.transpose(-1, 1)).view(B, S, N2, H, -1)  # [B, S, N, H, d]
        values = self.value_projection(gcn_values).view(B, S, N2, H, -1)

        queries = queries.transpose(2, 1)  # [B, N, L, H, d]
        keys = keys.transpose(2, 1)  # [B, N, S, H, d]
        values = values.transpose(2, 1)

        # Compute the attention
        new_values = self.inner_attention(queries, keys, values, attn_mask,
                                          query_lengths,
                                          key_lengths).view(B, N1, L, -1)

        new_values = new_values.transpose(2, 1)  # [B, L, N1, D]

        # Project the output and return
        return self.out_projection(new_values)


class TransEncoderLayer(nn.Module):
    def __init__(self,
                 attention,
                 d_model,
                 d_ff=None,
                 dropout=0.0,
                 activation="relu"):
        super(TransEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.TSA = temporalAttention(8, 8, 0.1)
        # self.Fusionlinear = nn.Conv1d(in_channels=2*d_model, out_channels=d_model, kernel_size=(
        #     1, 1))
        self.Fusionlinear = nn.Linear(2 * d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        # self.linear2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(
        #     1, 1))  # nn.Linear(d_ff, d_model)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D] [batch, time_steps, num_nodes, dimmension]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        # TriangularCausalMask(L, device=x.device)
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or LengthMask(x.new_full(
            (B,), L, dtype=torch.int64),
            device=x.device)
        # x [32,12,207,64] [batch, time_steps, num_nodes, dimmension]
        # Run self attention and add it to the input
        # x = x + self.dropout(
        #     self.attention(
        #         x,
        #         x,
        #         x,
        #         attn_mask=attn_mask,
        #         query_lengths=length_mask,
        #         key_lengths=length_mask,
        #         support=support,
        #     ))
        SSA = self.attention(x, x, x, attn_mask=attn_mask,
                             query_lengths=length_mask, key_lengths=length_mask, support=support)
        TSA = self.TSA(x)
        mix = torch.cat((SSA, TSA), dim=-1)
        x = x + self.dropout(self.Fusionlinear(mix))
        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        # y = self.dropout(self.activation(
        #     self.linear1(y.transpose(-1, 1))))  # [B, D, N, L]

        y = self.dropout(self.activation(self.linear1(y)))

        # y = self.dropout(self.linear2(y)).transpose(-1, 1)  # [B, L, N, D]
        # [B, L, N, D] TransformerFeedForward
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y)


class STLinearAttention(nn.Module):
    def __init__(self, feature_map=None, eps=1e-6):
        super(STLinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        # [B, N, L, H, d]
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, None, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("bnshd,bnshm->bnhmd", K, values)

        # Compute the normalizer
        Z = 1 / (torch.einsum("bnlhd,bnhd->bnlh", Q, K.sum(dim=2)) + self.eps)

        # Finally compute and return the new values
        V = torch.einsum("bnlhd,bnhmd,bnlh->bnlhm", Q, KV, Z)

        return V.contiguous()


class POLLAEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(POLLAEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, length_mask=None, support=None):
        # x [B, L, N, D] [32,12,207,64] ->? [521(207*3),32,64]
        B = x.shape[0]
        L = x.shape[1]
        N = x.shape[2]
        # TriangularCausalMask(L, device=x.device)
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or LengthMask(x.new_full(
            (B,), L, dtype=torch.int64),
            device=x.device)

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x,
                      attn_mask=attn_mask,
                      length_mask=length_mask,
                      support=support)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x


class polla_diff(nn.Module):
    def __init__(
            self,
            c_in,
            c_out,
            seq_len,
            out_len,
            adj_list,
            d_model=512,
            n_heads=8,
            n_layers=3,
            d_ff=8,
            nodes=207,
            aptinit=None,
            supports=None,
            order=2,
            dropout=0.0,
            activation="gelu",
            device=torch.device("cuda:0"),
    ):
        super(polla_diff, self).__init__()
        # encoding
        self.temporal_embedding = TemporalEncoding(d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.meter_embedding = nn.Linear(5, d_model)
        self.dropout = nn.Dropout(dropout)
        # self.temporal_attn = temporalAttention(8, 8)
        # GCN supports
        # self.supports = supports
        # support_len = 0 if supports is None else len(supports)
        # encoder

        self.encoder = POLLAEncoder([TransEncoderLayer(
            STSGCLdiffAttentionLayer(STLinearAttention(feature_map=elu_feature_map), d_model, n_heads, adj_list, nodes,
                                     order=order,
                                     dropout=dropout, history=seq_len), d_model, d_ff, dropout=dropout,
            activation=activation, ) for _ in range(n_layers)], norm_layer=torch.nn.LayerNorm(d_model))

        # output
        self.end_conv1 = nn.Conv2d(in_channels=seq_len,
                                   out_channels=out_len,
                                   kernel_size=(1, 1),
                                   bias=True)  # 降维
        self.end_conv2 = nn.Conv2d(in_channels=d_model,
                                   out_channels=c_out,
                                   kernel_size=(1, 1),
                                   bias=True)  # 降维
        self.LinearList = nn.ModuleList()
        for _ in range(out_len):
            self.LinearList.append(nn.Linear(d_model, c_out))
        # self.linear1 = nn.Linear(64, 64)
        self.finalLinear = nn.Linear(d_model, c_out)
        self.finalLinear2 = nn.Linear(seq_len, out_len)

    def forward(self,
                x,
                x_mark,
                y_mark,
                x_meter,
                attn_mask=None,
                length_mask=None,
                support=None):
        # x [BatchSize, Length, Nodes, Dimmesions]
        # support = self.supports
        x_meter = x_meter.unsqueeze(-2).repeat(1, 1, 34, 1)
        out = (self.value_embedding(x) + self.temporal_embedding(x_mark) +
               self.position_embedding(x))

        y_mark = self.temporal_embedding(y_mark)

        # x_meter = self.meterLinear(x_meter)

        out = self.dropout(out)
        out = self.encoder(out,
                           attn_mask=attn_mask,
                           length_mask=length_mask,
                           support=support)  # [B, L, N, D]
        # out = self.dropout(F.relu(self.linear1(out)))
        # out = out.permute(0,3,2,1)
        # x_meter_embedding = self.meter_embedding(x_meter)
        # out = torch.cat((out, x_meter_embedding), dim=-1)
        out = torch.tanh(self.finalLinear(out))
        out = out.permute(0, 3, 2, 1)
        out = self.finalLinear2(out)
        # out = F.relu(self.end_conv1(out))  # [B, OL, N, D]
        # out = self.end_conv2(out.transpose(-1,
        #                                    1)).transpose(-1,
        #                                                  1)  # [B, OL, N, OD]
        # OutHorizon = []

        # for i in range(out.shape[1]):
        #     OutHorizon.append(self.LinearList[i](out[:, i, :, :]))
        # out = self.finalLinear(out)
        # out = out.squeeze(-1)
        out = out.permute(0, 3, 2, 1)
        return out  # [B, L, N, D] -> [B,L,N]


"""
    FILE_TEST
"""
