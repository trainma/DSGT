import concurrent
from concurrent.futures import ThreadPoolExecutor

from sts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.AdaptiveGraphLearner import AdaptiveGraphLearner
from utils import construct_dynamic_adj_1hop, construct_adj_local, construct_dynamic_adj_1hop_torch2, \
    construct_dynamic_adj_lt, construct_adj_local_torch, \
    construct_dynamic_adj_1hop_torch, construct_dynamic_adj_lt_torch, construct_dynamic_adj_lt_optimized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class PTEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay, d_model, n_nodes):
        super(PTEmbedding, self).__init__()
        self.FC_pe = FC(
            input_dims=[64, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[n_nodes, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295
        self.dayofweek_emb = nn.Embedding(7, d_model)
        self.timeofday_emb = nn.Embedding(24 * 12, d_model)

    def forward(self, TE, PE, T=288):
        # spatial embedding
        # SE [325 64]->[1 1 325 64] TE[32 24 2]
        PE = PE.unsqueeze(0).unsqueeze(0)
        PE = self.FC_pe(PE)

        TE = TE.long()
        dayofweek_x = self.dayofweek_emb(TE[:, :, :, 0])
        timeofday_x = self.timeofday_emb(TE[:, :, :, 1])
        TE = dayofweek_x + timeofday_x

        return PE + TE


class SynorchouGraphAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(SynorchouGraphAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, SSE):
        batch_size = X.shape[0]
        X = torch.cat((X, SSE), dim=-1)
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
        # orginal K, change to batch_size
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, TPE=None):
        batch_size_ = X.shape[0]
        X = torch.cat((X, TPE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool).to(device)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        # orginal K, change to batch_size
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class SGTLayer(nn.Module):
    def __init__(self, K, d, bn_decay, dropout, mask=False, Satt=True, Tatt=True):
        super(SGTLayer, self).__init__()
        self.SGA = SynorchouGraphAttention(K, d, bn_decay)
        self.TEA = temporalAttention(K, d, bn_decay, mask=mask)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(K * d, K * d)
        self.dropout = nn.Dropout(dropout)
        self.gatedFusion = gatedFusion(K * d, bn_decay)
        self.Satt = Satt
        self.Tatt = Tatt

    def forward(self, X, STE, SSE):
        if self.Satt is True and self.Tatt is True:
            HS = self.SGA(X, SSE)
            HT = self.TEA(X, STE)
            H = self.gatedFusion(HS, HT)
            H = self.dropout(self.linear(H))
            del HS, HT
            return torch.add(X, H)

        elif self.Satt is False and self.Tatt is True:
            HT = self.TEA(X, STE)
            H = self.dropout(self.linear(HT))
            del HT
            return torch.add(X, H)

        elif self.Satt is True and self.Tatt is False:
            HS = self.SGA(X, SSE)
            H = self.dropout(self.linear(HS))
            del HS
            return torch.add(X, H)

        # HS = self.SGA(X, SSE)
        # HT = self.temporalAttention(X, STE)
        # H = self.gatedFusion(HS, HT)
        # H = self.dropout(self.linear(H))
        # del HS, HT
        # return torch.add(X, H)


class FusionAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, seq_len, pred_len):
        super(FusionAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)
        # self.adapative_conv = nn.Conv2d(in_channels=pred_len,
        #                            out_channels=seq_len,
        #                            kernel_size=(1, 1),
        #                            bias=True)  # 降维

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        # STE_pred = self.adapative_conv(STE_pred)

        # query = self.FC_q(X)
        # key = self.FC_k(STE_pred)
        # value = self.FC_v(STE_pred)
        query = self.FC_q(STE_pred)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


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


class SGT(nn.Module):
    def __init__(self, batch_size, adj_list, n_layers, K, d, n_hist, bn_decay, nodes, d_model, dropout, pred_len, Satt,
                 Tatt):
        super(SGT, self).__init__()
        D = K * d

        self.adjList = adj_list
        self.EDLinear = nn.Linear(6, d_model)
        self.MSTS_GCC = MSTS_GCC(
            batch_size, adj_list, n_hist, nodes, d_model, [d_model] * len(adj_list), len(adj_list), 'relu', True, True)
        self.PTEmbedding = PTEmbedding(D, bn_decay, d_model, nodes)
        self.Dynamic_graph_learner = AdaptiveGraphLearner(
            adj_list[0], n_hist, pred_len, d_model, dropout)
        self.Encoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, True, True) for _ in range(n_layers)])
        self.Decoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, True, True) for _ in range(n_layers)])
        self.transformAttention = FusionAttention(
            K, d, bn_decay, n_hist, pred_len)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[
            F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[
            F.relu, None], bn_decay=bn_decay)

        self.positional_encoding = PositionalEncoding(d_model=D)
        self.metrelogy_embedding = nn.Linear(5, d_model)
        self.end_conv = nn.Conv2d(
            in_channels=n_hist, out_channels=pred_len, kernel_size=(1, 1), bias=True)  # 降维
        self.adaptive_conv = nn.Conv2d(in_channels=pred_len,
                                       out_channels=n_hist,
                                       kernel_size=(1, 1),
                                       bias=True)  # 降维
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, x_mark, y_mark, x_meter, poi_embedding):
        x_meter = x_meter.unsqueeze(-2).repeat(1, 1, 34, 1)
        # X = torch.cat([X, x_meter], dim=-1)
        # ME = self.metrelogy_embedding(x_meter)
        x = torch.cat([x, x_meter], dim=-1)
        # X = torch.cat([X, ME], dim=-1)
        x = self.EDLinear(x)
        PE = self.positional_encoding(x)

        x = torch.add(x, PE)
        SSE = self.MSTS_GCC(x)
        PTE_his = self.PTEmbedding(x_mark, poi_embedding)
        PTE_pred = self.PTEmbedding(y_mark, poi_embedding)
        # PTE_pred = self.adaptive_conv(PTE_pred)
        # STE
        # STE = self.STEmbedding(self.SE, TE)
        # STE_his = STE[:, :self.num_his]  # [32 12 325 64]
        # STE_pred = STE[:, self.num_his:]  # [32 12 325 64]
        # encoder
        for net in self.Encoder:
            x = net(x, PTE_his, SSE, x_meter)
        # X = self.EDLinear(X)
        # transAtt
        # X = self.transformAttention(X, PTE_his, PTE_pred)
        x = self.transformAttention(x, PTE_his, PTE_pred)
        # decoder
        for net in self.Decoder:
            x = net(x, PTE_pred, SSE, x_meter)
        x = self.leaky_relu(self.end_conv(x))
        x = self.FC_2(x)
        return x


class AuxiliaryFeatureEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, c_meter=5, node_dim=34):
        super(AuxiliaryFeatureEmbedding, self).__init__()
        self.node_dim = node_dim
        # self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding2(d_model=d_model, embed_type=embed_type,
        #                                             freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #     d_model=d_model, embed_type=embed_type, freq=freq)
        self.temporal_embedding = TemporalEmbedding2(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.meterlogy_embedding = TokenEmbedding(
            c_in=c_meter, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_mark, x_meter):
        # node_dim = x.shape[-2]

        temporal_embedding = self.temporal_embedding(x_mark)
        temporal_embedding = torch.tile(
            temporal_embedding.unsqueeze(-2), (1, 1, self.node_dim, 1))
        meterology_embedding = self.meterlogy_embedding(x_meter)
        res = temporal_embedding + meterology_embedding
        return self.dropout(res)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        padding = 1
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv = nn.Conv2d(
            in_channels=c_in, out_channels=d_model, kernel_size=(1, 1), bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # [64 24 34 1] -> [64 1 24 34]
        x = self.tokenConv(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        return x


class DSGT(nn.Module):
    def __init__(self, batch_size, adj_list, n_layers, K, d, seq_len, bn_decay, nodes, d_model, dropout, pred_len, Satt,
                 Tatt, pre_graph, meter_flag, in_dim):
        super(DSGT, self).__init__()
        self.meter = meter_flag
        D = K * d
        self.pre_graph = pre_graph
        self.adjList = adj_list
        self.num_nodes = nodes
        self.EDLinear = nn.Linear(in_dim + 5, d_model)
        self.projection2 = nn.Linear(in_dim, d_model)
        self.Dynamic_graph_learner = AdaptiveGraphLearner(
            pre_graph, seq_len, in_dim, d_model, dropout)
        self.MSTS = MSTS_GCC_dynamic_graph(batch_size, seq_len, nodes, d_model,
                                           [d_model] *
                                           len(adj_list), len(
                adj_list), 'relu',
                                           True, True)
        self.PTEmbedding = PTEmbedding(D, bn_decay, d_model, nodes)
        # self.Encoder = nn.Moduledding(D, bn_decay, d_model, nodes)
        self.Encoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, True, True) for _ in range(n_layers)])
        self.Decoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, True, True) for _ in range(n_layers)])
        self.transformAttention = FusionAttention(
            K, d, bn_decay, seq_len, pred_len)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[
            F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[
            F.relu, None], bn_decay=bn_decay)

        self.positional_encoding = PositionalEncoding(d_model=D)
        self.meteorology_embedding = nn.Linear(5, d_model)
        self.end_conv = nn.Conv2d(
            in_channels=seq_len, out_channels=pred_len, kernel_size=(1, 1), bias=True)  # 降维
        self.adaptive_conv = nn.Conv2d(in_channels=pred_len,
                                       out_channels=seq_len,
                                       kernel_size=(1, 1),
                                       bias=True)  # 降维
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, y_mark, x_meter, poi_embedding):
        if self.meter == True:
            x_meter = x_meter.unsqueeze(-2).repeat(1, 1, self.num_nodes, 1)
            dynamic_graph_list = self.Graph_construct(x)
            x = torch.cat([x, x_meter], dim=-1)
            # X = torch.cat([X, ME], dim=-1)
            x = self.dropout(self.relu(self.EDLinear(x)))
            PE = self.positional_encoding(x)
            # E_M = self.meter_embedding(x_meter)
            x = torch.add(x, PE)
            # x = x + E_M
        if self.meter == False:
            dynamic_graph_list = self.Graph_construct(x)
            x = self.projection2(x)
            PE = self.positional_encoding(x)
            # E_M = self.meteorology_embedding(x_meter)
            x = torch.add(x, PE)
            # x = torch.add(x, E_M)
        E_SP = self.MSTS(x, dynamic_graph_list)
        E_AU_his = self.PTEmbedding(x_mark, poi_embedding)
        E_AU_fut = self.PTEmbedding(y_mark, poi_embedding)

        for net in self.Encoder:
            x = net(x, E_AU_his, E_SP)

        x = self.transformAttention(x, E_AU_his, E_AU_fut)

        for net in self.Decoder:
            x = net(x, E_AU_fut, E_SP)
        x = self.relu(self.end_conv(x))
        x = self.FC_2(x)
        return x

    # def Graph_construct(self, x):
    #     adj_graph = self.Dynamic_graph_learner(x)
    #     adj_graph2 = adj_graph.clone()
    #     adj_graph3 = adj_graph.clone()
    #     B = adj_graph.shape[0]
    #     N = adj_graph.shape[1]
    #     adj_graph_local = torch.zeros((B, 3*N, 3*N))
    #     adj_graph_1hop = torch.zeros((B, 3*N, 3*N))
    #     adj_graph_2hop = torch.zeros((B, 3*N, 3*N))
    #
    #     def process_graph(i):
    #         adj_graph_local[i] = construct_adj_local_torch(adj_graph[i], 3)# 你需要定义这个函数
    #         adj_graph_1hop[i] = construct_dynamic_adj_1hop_torch2(adj_graph2[i], 3)
    #         adj_graph_2hop[i] = construct_dynamic_adj_lt_optimized(adj_graph3[i], 3)
    #
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         executor.map(process_graph, range(B))
    #
    #     return [adj_graph_local, adj_graph_1hop, adj_graph_2hop]

    def Graph_construct(self, x):
        adj_graph = self.Dynamic_graph_learner(x)
        adj_graph2 = adj_graph.clone()
        adj_graph3 = adj_graph.clone()
        B = adj_graph.shape[0]
        N = adj_graph.shape[1]
        adj_graph_local = torch.zeros((B, 3 * N, 3 * N))
        adj_graph_1hop = torch.zeros((B, 3 * N, 3 * N))
        adj_graph_2hop = torch.zeros((B, 3 * N, 3 * N))
        for i in range(B):
            adj_graph_local[i] = construct_adj_local_torch(adj_graph[i], 3)
            adj_graph_1hop[i] = construct_dynamic_adj_1hop_torch2(
                adj_graph2[i], 3)
            adj_graph_2hop[i] = construct_dynamic_adj_lt_optimized(
                adj_graph3[i], 3)
        return [adj_graph_local, adj_graph_1hop, adj_graph_2hop]


class SGT_noSTS(nn.Module):
    def __init__(self, batch_size, adjList, n_layers, K, d, seq_len, bn_decay, nodes, d_model, dropout, out_len, Satt,
                 Tatt, Fatt):
        super(SGT_noSTS, self).__init__()
        D = K * d
        self.Fatt = Fatt
        self.num_his = seq_len
        self.adjList = adjList
        self.Synorchous_Sptial_embedding = MSTS_GCC(batch_size,
                                                    adjList, seq_len, nodes, d_model, [
                                                        d_model] * 3, 3, 'relu', True,
                                                    True)

        self.PTEmbedding = PTEmbedding(D, bn_decay, d_model, nodes)
        self.Encoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, Satt, Tatt) for _ in range(n_layers)])
        self.Decoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, Satt, Tatt) for _ in range(n_layers)])
        self.transformAttention = FusionAttention(
            K, d, bn_decay, seq_len, out_len)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D],
                       activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[
            F.relu, None], bn_decay=bn_decay)
        # self.temporal_encoding = TemporalEncoding(d_model=D)
        self.EDLinear = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model=D)
        self.metrelogy_embedding = nn.Linear(5, d_model)
        self.end_conv = nn.Conv2d(in_channels=seq_len,
                                  out_channels=out_len,
                                  kernel_size=(1, 1),
                                  bias=True)  # 降维
        self.adaptive_conv = nn.Conv2d(in_channels=out_len,
                                       out_channels=seq_len,
                                       kernel_size=(1, 1),
                                       bias=True)  # 降维
        self.relu = nn.ReLU()

    def forward(self, X, x_mark, y_mark, x_meter, poiE):
        x_meter = x_meter.unsqueeze(-2).repeat(1, 1, 34, 1)
        # X = torch.cat([X, x_meter], dim=-1)
        # ME = self.metrelogy_embedding(x_meter)
        X = self.FC_1(X)
        PE = self.positional_encoding(X)

        X = torch.add(X, PE)
        SSE = self.Synorchous_Sptial_embedding(X)
        PTE_his = self.PTEmbedding(x_mark, poiE)
        PTE_pred = self.PTEmbedding(y_mark, poiE)
        # PTE_pred = self.adaptive_conv(PTE_pred)
        # STE
        # STE = self.STEmbedding(self.SE, TE)
        # STE_his = STE[:, :self.num_his]  # [32 12 325 64]
        # STE_pred = STE[:, self.num_his:]  # [32 12 325 64]
        # encoder
        for net in self.Encoder:
            X = net(X, PTE_his, SSE, x_meter)

        # transAtt
        if self.Fatt == True:
            X = self.transformAttention(X, PTE_his, PTE_pred)
        else:
            X = self.EDLinear(X)
        # decoder
        for net in self.Decoder:
            X = net(X, PTE_pred, SSE, x_meter)
        X = self.relu(self.end_conv(X))
        X = self.FC_2(X)
        return X


class SGT_no_multi_scale(nn.Module):
    def __init__(self, batch_size, adjList, n_layers, K, d, seq_len, bn_decay, nodes, d_model, dropout, out_len, Satt,
                 Tatt, Fatt):
        super(SGT_no_multi_scale, self).__init__()
        D = K * d
        self.Fatt = Fatt
        self.num_his = seq_len
        self.adjList = adjList

        self.Synorchous_Sptial_embedding = MULTI_STS3(batch_size,
                                                      adjList, seq_len, nodes, d_model, [
                                                          d_model] * 3, 3, 'relu', True,
                                                      True)
        self.PTEmbedding = PTEmbedding(D, bn_decay, d_model, nodes)
        self.Encoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, Satt, Tatt) for _ in range(n_layers)])
        self.Decoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, Satt, Tatt) for _ in range(n_layers)])
        self.transformAttention = FusionAttention(
            K, d, bn_decay, seq_len, out_len)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D],
                       activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[
            F.relu, None], bn_decay=bn_decay)
        # self.temporal_encoding = TemporalEncoding(d_model=D)
        self.EDLinear = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model=D)
        self.metrelogy_embedding = nn.Linear(5, d_model)
        self.end_conv = nn.Conv2d(in_channels=seq_len,
                                  out_channels=out_len,
                                  kernel_size=(1, 1),
                                  bias=True)  # 降维
        self.adaptive_conv = nn.Conv2d(in_channels=out_len,
                                       out_channels=seq_len,
                                       kernel_size=(1, 1),
                                       bias=True)  # 降维
        self.relu = nn.ReLU()

    def forward(self, X, x_mark, y_mark, x_meter, poiE):
        x_meter = x_meter.unsqueeze(-2).repeat(1, 1, 34, 1)
        # X = torch.cat([X, x_meter], dim=-1)
        # ME = self.metrelogy_embedding(x_meter)
        X = self.FC_1(X)
        PE = self.positional_encoding(X)

        X = torch.add(X, PE)
        SSE = self.Synorchous_Sptial_embedding(X)
        PTE_his = self.PTEmbedding(x_mark, poiE)
        PTE_pred = self.PTEmbedding(y_mark, poiE)
        # PTE_pred = self.adaptive_conv(PTE_pred)
        # STE
        # STE = self.STEmbedding(self.SE, TE)
        # STE_his = STE[:, :self.num_his]  # [32 12 325 64]
        # STE_pred = STE[:, self.num_his:]  # [32 12 325 64]
        # encoder
        for net in self.Encoder:
            X = net(X, PTE_his, SSE, x_meter)

        # transAtt
        if self.Fatt:
            X = self.transformAttention(X, PTE_his, PTE_pred)
        else:
            X = self.EDLinear(X)
        # decoder
        for net in self.Decoder:
            X = net(X, PTE_pred, SSE, x_meter)
        X = self.relu(self.end_conv(X))
        X = self.FC_2(X)
        return X
