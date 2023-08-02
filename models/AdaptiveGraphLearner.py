import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        self.padding_size = math.ceil(
            kernel_size) if padding == 'SAME' else [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1],
                       self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class GraphAttention(nn.Module):

    def __init__(self, K, d, bn_decay):
        super(GraphAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=D, units=D, activations=F.leaky_relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.leaky_relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.leaky_relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.leaky_relu,
                     bn_decay=bn_decay)

    def forward(self, X):
        batch_size = X.shape[0]
        # X = torch.cat((X, SSE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        # original K, change to batch_size
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class DynamicLearner(nn.Module):
    def __init__(self, n_hist: int, n_in: int, node_dim: int, dropout: float):
        super(DynamicLearner, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nodes = nn.Sequential(
            nn.Conv2d(n_in, node_dim, kernel_size=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(node_dim, node_dim, kernel_size=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(node_dim, node_dim, kernel_size=(1, n_hist))
        )
        self.attention = GraphAttention(K=node_dim // 8, d=8, bn_decay=0.1)

    def forward(self, inputs: Tensor, mask=None):

        x = inputs.transpose(1, 3)
        nodes = self.nodes(x)

        nodes = nodes.transpose(1, 3)
        nodes = self.attention(nodes)

        matrix = nodes.squeeze(1)

        self.dropout(matrix)
        A_mi = torch.einsum('bud,bvd->buv', [matrix, matrix])
        A_mi = torch.mul(A_mi, mask)
        # normalization
        A_mi = F.normalize(A_mi, p=2, dim=-1)
        return A_mi


class AdaptiveGraphLearner(nn.Module):
    def __init__(self, pre_graph: Tensor, n_hist: int, n_in: int, node_dim: int, dropout: float):
        super(AdaptiveGraphLearner, self).__init__()
        # self.adaptive = nn.Parameter(supports, requires_grad=learn_macro)
        self.pre_graph = pre_graph
        self.mi_learner = DynamicLearner(n_hist, n_in, node_dim, dropout)

    def forward(self, inputs: Tensor = None) -> Tensor:
        pre_graph = self.pre_graph

        mask = torch.where(pre_graph > 0, torch.ones_like(
            pre_graph), torch.zeros_like(pre_graph))
        dynamic_graph = pre_graph.unsqueeze(0) + self.mi_learner(inputs, mask)

        return F.normalize(torch.relu(dynamic_graph), p=2, dim=-1)
