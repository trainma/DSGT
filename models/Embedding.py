from torch import nn
import torch
import math
import torch.nn.functional as F
from models.Attention import FC


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


class PTEmbedding(nn.Module):

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

    def forward(self, TE, PE):

        PE = PE.unsqueeze(0).unsqueeze(0)
        PE = self.FC_pe(PE)

        TE = TE.long()
        dayofweek_x = self.dayofweek_emb(TE[:, :, :, 0])
        timeofday_x = self.timeofday_emb(TE[:, :, :, 1])
        TE = dayofweek_x + timeofday_x

        return PE + TE

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding2(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding2, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)


    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 0])

        return minute_x + hour_x + weekday_x

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