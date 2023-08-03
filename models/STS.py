import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation="relu"):
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
        adj = self.adj
        # if mask is not None:
        #     adj = adj.to(mask.device) * mask

        x = torch.einsum("nm, mbc->nbc", adj.to(x.device), x)  # 3*N, B, Cin
        # Add batch dimension
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


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_vertices, activation="relu"):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {"GLU", "relu"}

        if self.activation == "GLU":
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None, batch_graph=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :param batch_graph: (E,3*N,3*N)
        :return: (3*N, B, Cout)
        """
        adj = batch_graph
        # if mask is not None:
        #     adj = adj.to(mask.device) * mask
        # No batch dimension
        # x = torch.einsum("nm, mbc->nbc", adj.to(x.device), x)  # 3*N, B, Cin
        # Add batch dimension
        x = torch.einsum("bnm, mbc->nbc", adj.to(x.device), x)  # 3*N, B, Cin

        if self.activation == "GLU":
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim,
                                   dim=-1)  # 3*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs
            return out

        elif self.activation == "relu":
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class ST_block(nn.Module):
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
        super(ST_block, self).__init__()
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

    def forward(self, x, mask=None, ):
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


class ST_block2(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dims,
                 num_of_vertices,
                 activation="GLU"):
        """
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(ST_block2, self).__init__()
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            GCN(
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation,
            ))

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                GCN(
                    in_dim=self.out_dims[i - 1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))

    def forward(self, x, mask=None, batch_graph=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :param batch_graph:(B 3N 3N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask, batch_graph)
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


class Multi_scale_graph_fusion(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super(Multi_scale_graph_fusion, self).__init__()
        self.param1 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.param2 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.param3 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bfc = nn.Parameter(torch.randn(batch_size, out_dim))

    def forward(self, x, y, z):
        """
        :param x: (N, B, 64)
        :param y: (N, B, 64)
        :param z: (N, B, 64)
        :return: (N, B, 64)
        """
        x = torch.einsum("vbd,dj->vbj", x, self.param1)
        y = torch.einsum("vbd,dj->vbj", y, self.param2)
        z = torch.einsum("vbd,dj->vbj", z, self.param3)
        # return torch.tanh(x + y + z + self.bfc)
        return torch.sigmoid(x + y + z + self.bfc)


class two_scale_graph_fusion(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super(two_scale_graph_fusion, self).__init__()
        self.param1 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.param2 = nn.Parameter(torch.randn(in_dim, out_dim))
        # self.param3 = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bfc = nn.Parameter(torch.randn(batch_size, out_dim))

    def forward(self, x, y):
        """
        :param x: (N, B, 64)
        :param y: (N, B, 64)
        :return: (N, B, 64)
        """
        x = torch.einsum("vbd,dj->vbj", x, self.param1)
        y = torch.einsum("vbd,dj->vbj", y, self.param2)
        # z = torch.einsum("vbd,dj->vbj", z, self.param3)
        # return torch.tanh(x + y + z + self.bfc)
        return torch.sigmoid(x + y + self.bfc)


class MSTS_GCC(nn.Module):
    def __init__(
            self,
            batch_size,
            adjList,
            history,
            num_of_vertices,
            in_dim,
            out_dims,
            strides=3,
            activation="relu",
            temporal_emb=True,
            spatial_emb=True,
    ):
        """
        :param adjList: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}

        """
        super(MSTS_GCC, self).__init__()
        self.adjList = adjList
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.temporal_STSGCMS = nn.ModuleList()
        self.spatial_temporal_STSGCMS = nn.ModuleList()
        self.long_spatial_temporal_STSGCMS = nn.ModuleList()
        self.LinearList1 = nn.ModuleList()
        self.Fusion = Multi_scale_graph_fusion(
            batch_size=batch_size, in_dim=self.in_dim, out_dim=self.in_dim)


        for _ in range(3):
            self.LinearList1.append(
                nn.Linear(self.out_dims[0], self.out_dims[0]))
        self.OutLinear = nn.Linear(self.out_dims[0] * 3, self.out_dims[0])

        for _ in range(self.history):
            self.temporal_STSGCMS.append(
                ST_block(
                    adj=self.adjList[0],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))
        for _ in range(self.history):
            self.spatial_temporal_STSGCMS.append(
                ST_block(
                    adj=self.adjList[1],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))
        for _ in range(self.history):
            self.long_spatial_temporal_STSGCMS.append(
                ST_block(
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

    def forward(self, x, adj_list=None, mask=None):
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

            t1 = torch.reshape(t,
                               shape=[
                                   batch_size,
                                   self.strides * self.num_of_vertices,
                                   self.in_dim
                               ])
            t2 = t1.clone()
            t3 = t1.clone()

            # (3*N, B, Cin) -> (N, B, Cout)
            t1 = self.temporal_STSGCMS[i](t1.permute(1, 0, 2), mask)
            t2 = self.spatial_temporal_STSGCMS[i](t2.permute(1, 0, 2), mask)
            t3 = self.long_spatial_temporal_STSGCMS[i](t3.permute(1, 0, 2), mask)
            # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            t1 = F.relu(self.LinearList1[0](t1))
            t2 = F.relu(self.LinearList1[1](t2))
            t3 = F.relu(self.LinearList1[2](t3))

            test = self.Fusion(t1, t2, t3)
            test = torch.unsqueeze(test.permute(1, 0, 2), dim=1)
            need_concat.append(test)


        out = torch.cat(need_concat, dim=1)
        del need_concat, batch_size
        return out


class MSTS_GCC_dynamic_graph(nn.Module):
    def __init__(
            self,
            batch_size,
            history,
            num_of_vertices,
            in_dim,
            out_dims,
            strides=3,
            activation="relu",
            temporal_emb=True,
            spatial_emb=True,
    ):
        """
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}

        """
        super(MSTS_GCC_dynamic_graph, self).__init__()
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.local_sts_block = nn.ModuleList()
        self.hop1_sts_block = nn.ModuleList()
        self.hop2_sts_block = nn.ModuleList()
        self.LinearList1 = nn.ModuleList()
        self.Fusion = Multi_scale_graph_fusion(
            batch_size=batch_size, in_dim=self.in_dim, out_dim=self.in_dim)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
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
            self.local_sts_block.append(
                ST_block2(
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))
        for _ in range(self.history):
            self.hop1_sts_block.append(
                ST_block2(
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))
        for _ in range(self.history):
            self.hop2_sts_block.append(
                ST_block2(
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

    def forward(self, x, adj_list=None, mask=None):

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
            t1 = self.local_sts_block[i](t1.permute(1, 0, 2), mask, adj_list[0])
            t2 = self.hop1_sts_block[i](t2.permute(1, 0, 2), mask, adj_list[1])
            t3 = self.hop2_sts_block[i](t3.permute(1, 0, 2), mask, adj_list[2])
            # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            t1 = self.leaky_relu(self.LinearList1[0](t1))
            t2 = self.leaky_relu(self.LinearList1[1](t2))
            t3 = self.leaky_relu(self.LinearList1[2](t3))

            test = self.Fusion(t1, t2, t3)
            test = torch.unsqueeze(test.permute(1, 0, 2), dim=1)
            need_concat.append(test)

            # t = torch.cat((t1, t2, t3), dim=-1)
            # t = self.OutLinear(t)  # (27 128 64)
            # t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)
            # need_concat.append(t)

        # modified (B, T-2, N, Cout) -> (B, T, N, Cout)
        out = torch.cat(need_concat, dim=1)
        del need_concat, batch_size
        return out


class MULTI_STS2(nn.Module):
    def __init__(
            self,
            batch_size,
            adjList,
            history,
            num_of_vertices,
            in_dim,
            out_dims,
            strides=3,
            activation="relu",
            temporal_emb=True,
            spatial_emb=True,
    ):

        super(MULTI_STS2, self).__init__()
        self.adjList = adjList
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.temporal_STSGCMS = nn.ModuleList()
        self.spatial_temporal_STSGCMS = nn.ModuleList()
        self.long_spatial_temporal_STSGCMS = nn.ModuleList()
        self.LinearList1 = nn.ModuleList()
        self.Fusion = two_scale_graph_fusion(batch_size=batch_size, in_dim=64, out_dim=64)
        # self.Fusion = Multi_scale_graph_fusion(
        #     batch_size=batch_size, in_dim=64, out_dim=64)
        for _ in range(2):
            self.LinearList1.append(
                nn.Linear(self.out_dims[0], self.out_dims[0]))
        self.OutLinear = nn.Linear(self.out_dims[0] * 2, self.out_dims[0])
        for _ in range(self.history):
            self.temporal_STSGCMS.append(
                ST_block(
                    adj=self.adjList[0],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                ))

        for _ in range(self.history):
            self.spatial_temporal_STSGCMS.append(
                ST_block(
                    adj=self.adjList[1],
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
            t2 = t1.clone()

            # (3*N, B, Cin) -> (N, B, Cout)
            t1 = self.temporal_STSGCMS[i](t1.permute(1, 0, 2), mask)
            t2 = self.spatial_temporal_STSGCMS[i](t2.permute(1, 0, 2), mask)
            # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            t1 = F.relu(self.LinearList1[0](t1))
            t2 = F.relu(self.LinearList1[1](t2))
            # F5F5F5
            test = self.Fusion(t1, t2)
            test = torch.unsqueeze(test.permute(1, 0, 2), dim=1)
            need_concat.append(test)

            # t = torch.cat((t1, t2, t3), dim=-1)
            # t = self.OutLinear(t)  # (27 128 64)
            # t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)
            # need_concat.append(t)

        # modified (B, T-2, N, Cout) -> (B, T, N, Cout)
        out = torch.cat(need_concat, dim=1)
        del need_concat, batch_size
        return out


class MULTI_STS3(nn.Module):
    def __init__(
            self,
            batch_size,
            adjList,
            history,
            num_of_vertices,
            in_dim,
            out_dims,
            strides=3,
            activation="relu",
            temporal_emb=True,
            spatial_emb=True,
    ):

        super(MULTI_STS3, self).__init__()
        self.adjList = adjList
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.temporal_STSGCMS = nn.ModuleList()
        self.spatial_temporal_STSGCMS = nn.ModuleList()
        self.long_spatial_temporal_STSGCMS = nn.ModuleList()
        self.LinearList1 = nn.ModuleList()
        self.Fusion = two_scale_graph_fusion(batch_size=batch_size, in_dim=64, out_dim=64)
        # self.Fusion = Multi_scale_graph_fusion(
        #     batch_size=batch_size, in_dim=64, out_dim=64)
        for _ in range(2):
            self.LinearList1.append(
                nn.Linear(self.out_dims[0], self.out_dims[0]))
        self.OutLinear = nn.Linear(self.out_dims[0] * 2, self.out_dims[0])
        for _ in range(self.history):
            self.temporal_STSGCMS.append(
                ST_block(
                    adj=self.adjList[0],
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

            t1 = torch.reshape(t,
                               shape=[
                                   batch_size,
                                   self.strides * self.num_of_vertices,
                                   self.in_dim
                               ])

            t1 = self.temporal_STSGCMS[i](t1.permute(1, 0, 2), mask)
            # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            t1 = F.relu(self.LinearList1[0](t1))
            test = torch.unsqueeze(t1.permute(1, 0, 2), dim=1)
            need_concat.append(test)

        out = torch.cat(need_concat, dim=1)
        del need_concat, batch_size
        return out
