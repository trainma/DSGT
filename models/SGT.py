import concurrent
from concurrent.futures import ThreadPoolExecutor

from models.Attention import gatedFusion, FC, SynorchouGraphAttention, temporalAttention, EncoderDecoderAttention
from models.Embedding import PositionalEncoding, PTEmbedding
from sts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AdaptiveGraphLearner import AdaptiveGraphLearner
from utils import construct_dynamic_adj_1hop, construct_adj_local, construct_dynamic_adj_1hop_torch2, \
    construct_dynamic_adj_lt, construct_adj_local_torch, \
    construct_dynamic_adj_1hop_torch, construct_dynamic_adj_lt_torch, construct_dynamic_adj_lt_optimized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        self.transformAttention = EncoderDecoderAttention(
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





class DSGT(nn.Module):
    def __init__(self, batch_size, adj_list, n_layers, K, d, seq_len, bn_decay, nodes, d_model, dropout, pred_len, Satt,
                 Tatt, pre_graph, meter_flag, in_dim, spatial_embed, temporal_embed):
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
        self.MSTS_GCC = MSTS_GCC_dynamic_graph(batch_size, seq_len, nodes, d_model,
                                               [d_model] *
                                               len(adj_list), len(
                adj_list), 'relu', True, True)
        self.PTEmbedding = PTEmbedding(D, bn_decay, d_model, nodes)
        # self.Encoder = nn.Moduledding(D, bn_decay, d_model, nodes)
        self.Encoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, spatial_embed, temporal_embed) for _ in range(n_layers)])
        self.Decoder = nn.ModuleList(
            [SGTLayer(K, d, bn_decay, dropout, False, spatial_embed, temporal_embed) for _ in range(n_layers)])
        self.transformAttention = EncoderDecoderAttention(
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
        E_SP = self.MSTS_GCC(x, dynamic_graph_list)
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
        self.transformAttention = EncoderDecoderAttention(
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
        self.transformAttention = EncoderDecoderAttention(
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
