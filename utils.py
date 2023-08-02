import os
import torch
import random
import argparse
import itertools
import numpy as np
import pandas as pd
from torch.utils import data

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


"""图相关"""


def get_adjacency_matrix(distance_df_filename,
                         num_of_vertices,
                         type_='connectivity',
                         id_filename=None):
    """
    :param distance_df_filename: str, csv边信息文件路径
    :param num_of_vertices:int, 节点数量
    :param type_:str, {connectivity, distance}
    :param id_filename:str 节点信息文件， 有的话需要构建字典
    """
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)  # 构建临接矩阵

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {
                int(i): idx
                for idx, i in enumerate(f.read().strip().split('\n'))
            }  # 建立映射列表
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1
        return A

    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type_ == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")

    return A


def construct_adj_local(A, steps):
    """
    构建local 时空图
    :param A: np.ndarray, adjacency matrix, shape is (N, N)
    :param steps: 选择几个时间步来构建图
    :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """
    N = len(A)  # 获得行数
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        """对角线代表各个时间步自己的空间图，也就是A"""
        adj[i * N:(i + 1) * N, i * N:(i + 1) * N] = A
        # adj[i * N:(i + 1) * N, (i+1)*N:(i+2)*N] = A
        # adj[(i + 1) * N:(i + 2) * N, i * N:(i + 1) * N] = A

        # adj[i * N:(i+1) * N, (i + 1) * N:(i + 2) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            """每个节点只会连接相邻时间步的自己"""
            weight = (np.sum(A[i, :])) / (N - 1)
            adj[k * N + i, (k + 1) * N + i] = weight
            adj[(k + 1) * N + i, k * N + i] = weight

    # for j in range(N):
    #     for k in range(steps-1):

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1

    return adj


# def multi_adj_construct(batchAdjList, steps):
#     B, L, N, N = batchAdjList.shape
#
#     ConstructAdj = np.zeros((B, L, N * 3, N * 3))
#     ConstructAdj_oneHop = np.zeros((B, L, N * 3, N * 3))
#
#     for batch in range(batchAdjList.shape[0]):
#         graphs = batchAdjList[batch]
#         padding_before = np.expand_dims(graphs[0], 0)
#         padding_last = np.expand_dims(graphs[-1], 0)
#         ConcatGraphs = np.concatenate((padding_before, graphs, padding_last),
#                                       axis=0)
#
#         for step in range(steps):
#             ConstructAdj[batch,
#                          step] = construct_dynamic_adj(ConcatGraphs[step],
#                                                        ConcatGraphs[step + 1],
#                                                        ConcatGraphs[step + 2],
#                                                        3)
#             # ConstructAdj_oneHop[batch, step] = construct_dynamic_adj_1hop(
#             #     ConcatGraphs[step], ConcatGraphs[step+1], ConcatGraphs[step+2], 3)
#     return ConstructAdj


def construct_dynamic_adj_lt(A1, steps):
    N = len(A1)  # 获得行数
    adj = np.zeros((N * steps, N * steps))
    temp = np.zeros((N, N))
    adj[0 * N:(0 + 1) * N, 0 * N:(0 + 1) * N] = A1
    adj[1 * N:(1 + 1) * N, 1 * N:(1 + 1) * N] = A1
    adj[2 * N:(2 + 1) * N, 2 * N:(2 + 1) * N] = A1

    for row in range(N):
        for col in range(N):
            if row != col and adj[row, col] > 0:
                for inner_col in range(N):
                    if row != inner_col and inner_col != col and adj[col, inner_col] > 0:
                        temp[row, inner_col] = np.sum(
                            A1[row, :]) / (np.sum(A1[row, :] > 0))

    adj[1 * N:(1 + 1) * N, 0 * N:(0 + 1) * N] = temp
    adj[0 * N:(0 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[2 * N:(2 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[1 * N:(1 + 1) * N, 2 * N:(2 + 1) * N] = temp

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1

    return adj


def construct_dynamic_adj_1hop(A1, steps):
    N = len(A1)  # 获得行数
    adj = np.zeros((N * steps, N * steps))
    temp = np.zeros((N, N))
    adj[0 * N:(0 + 1) * N, 0 * N:(0 + 1) * N] = A1
    adj[1 * N:(1 + 1) * N, 1 * N:(1 + 1) * N] = A1
    adj[2 * N:(2 + 1) * N, 2 * N:(2 + 1) * N] = A1

    for row in range(N):
        for col in range(N):
            if row != col and A1[row, col] > 0:
                temp[row, col] = np.sum(A1[row, :]) / (np.sum(A1[row, :] > 0))

    adj[1 * N:(1 + 1) * N, 0 * N:(0 + 1) * N] = temp
    adj[0 * N:(0 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[2 * N:(2 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[1 * N:(1 + 1) * N, 2 * N:(2 + 1) * N] = temp

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1
    return adj


def construct_adj_local(A, steps):
    N = len(A)  # 获得行数
    adj = np.zeros((N * steps, N * steps))
    temp = np.zeros((N, N))
    adj[0 * N:(0 + 1) * N, 0 * N:(0 + 1) * N] = A
    adj[1 * N:(1 + 1) * N, 1 * N:(1 + 1) * N] = A
    adj[2 * N:(2 + 1) * N, 2 * N:(2 + 1) * N] = A

    for row, col in itertools.product(range(N), range(N)):
        if row == col and A[row, col] > 0:
            temp[row, col] = np.sum(A[row, :]) / (np.sum(A[row, :] > 0))

    adj[1 * N:(1 + 1) * N, 0 * N:(0 + 1) * N] = temp
    adj[0 * N:(0 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[2 * N:(2 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[1 * N:(1 + 1) * N, 2 * N:(2 + 1) * N] = temp

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1
    return adj


def construct_adj_local_torch(A, steps):
    N = A.shape[1]  # 获得行数
    adj = torch.zeros((N * steps, N * steps))
    temp = torch.zeros((N, N))
    adj[0 * N:(0 + 1) * N, 0 * N:(0 + 1) * N] = A
    adj[1 * N:(1 + 1) * N, 1 * N:(1 + 1) * N] = A
    adj[2 * N:(2 + 1) * N, 2 * N:(2 + 1) * N] = A

    for row, col in itertools.product(range(N), range(N)):
        if row == col and A[row, col] > 0:
            temp[row, col] = torch.sum(
                A[row, :]) / (torch.sum(A[row, :] > 0))

    adj[1 * N:(1 + 1) * N, 0 * N:(0 + 1) * N] = temp
    adj[0 * N:(0 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[2 * N:(2 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[1 * N:(1 + 1) * N, 2 * N:(2 + 1) * N] = temp

    for i in range(len(adj)):
        """add self-loop"""
        adj[i, i] = 1
    return adj


def construct_adj_local_torch2(A, steps):
    B = A.shape[0]
    N = A.shape[1]  # 获得行数
    graph = torch.zeros((B, N * steps, N * steps))
    for i in range(B):
        adj = torch.zeros((N * steps, N * steps))
        temp = torch.zeros((N, N))
        adj[0 * N:(0 + 1) * N, 0 * N:(0 + 1) * N] = A[i]
        adj[1 * N:(1 + 1) * N, 1 * N:(1 + 1) * N] = A[i]
        adj[2 * N:(2 + 1) * N, 2 * N:(2 + 1) * N] = A[i]

        # 将product函数替换为矩阵乘法
        temp = A[i] * torch.sum(A[i], dim=1, keepdim=True) / \
               (torch.sum(A[i] > 0, dim=1, keepdim=True) + 1e-8)
        adj[1 * N:(1 + 1) * N, 0 * N:(0 + 1) * N] = temp
        adj[0 * N:(0 + 1) * N, 1 * N:(1 + 1) * N] = temp
        adj[2 * N:(2 + 1) * N, 1 * N:(1 + 1) * N] = temp
        adj[1 * N:(1 + 1) * N, 2 * N:(2 + 1) * N] = temp

        # 将for循环替换为矩阵运算
        for j in range(len(adj)):
            """add self-loop"""
            adj[j, j] = 1
        graph[i] = adj

    return graph


def construct_dynamic_adj_lt_torch(A1, steps):
    N = len(A1)  # 获得行数
    adj = torch.zeros((N * steps, N * steps))
    temp = torch.zeros((N, N))

    # 将A1复制到adj矩阵中
    for i in range(steps):
        adj[i * N:(i + 1) * N, i * N:(i + 1) * N] = A1

    # 对temp矩阵进行赋值
    for row in range(N):
        for col in range(N):
            if row != col and adj[row, col] > 0:
                for inner_col in range(N):
                    if row != inner_col and inner_col != col and adj[col, inner_col] > 0:
                        temp[row, inner_col] = torch.sum(
                            A1[row, :]) / (torch.sum(A1[row, :] > 0))

    # 将temp矩阵复制到adj矩阵中
    for i in range(steps - 1):
        adj[(i + 1) * N:(i + 2) * N, i * N:(i + 1) * N] = temp
        adj[i * N:(i + 1) * N, (i + 1) * N:(i + 2) * N] = temp

    # 添加自环
    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def construct_dynamic_adj_lt_optimized(A1, steps):
    N = len(A1)
    adj = torch.zeros((N * steps, N * steps))
    temp = torch.where(A1 > 0, torch.sum(A1, axis=1, keepdim=True).T / (torch.sum(A1 > 0, axis=1, keepdim=True).T),
                       torch.zeros(A1.shape).to(A1.device))

    adj[0:N, 0:N] = A1
    for i in range(1, steps):
        adj[i * N:(i + 1) * N, (i - 1) * N:i * N] = temp
        adj[(i - 1) * N:i * N, i * N:(i + 1) * N] = temp

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def construct_dynamic_adj_1hop_torch(A, steps):
    N = len(A)  # 获得行数
    adj = torch.zeros((N * steps, N * steps))
    temp = torch.zeros((N, N))
    adj[0 * N:(0 + 1) * N, 0 * N:(0 + 1) * N] = A
    adj[1 * N:(1 + 1) * N, 1 * N:(1 + 1) * N] = A
    adj[2 * N:(2 + 1) * N, 2 * N:(2 + 1) * N] = A

    for row in range(N):
        for col in range(N):
            if row != col and A[row, col] > 0:
                temp[row, col] = torch.sum(
                    A[row, :]) / (torch.sum(A[row, :] > 0))

    adj[1 * N:(1 + 1) * N, 0 * N:(0 + 1) * N] = temp
    adj[0 * N:(0 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[2 * N:(2 + 1) * N, 1 * N:(1 + 1) * N] = temp
    adj[1 * N:(1 + 1) * N, 2 * N:(2 + 1) * N] = temp

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1
    return adj


def construct_dynamic_adj_1hop_torch2(A, steps):
    N = len(A)  # 获得行数
    adj = torch.zeros((N * steps, N * steps))
    temp = torch.zeros((N, N))  # 获得行数

    # 优化后
    for row in range(N):
        temp[row, :] = torch.sum(A[row, :]) / (torch.sum(A[row, :] > 0))

    for i in range(steps):
        adj[i * N:(i + 1) * N, i * N:(i + 1) * N] = A
        if i > 0:
            adj[i * N:(i + 1) * N, (i - 1) * N:(i - 1 + 1) * N] = temp
            adj[(i - 1) * N:(i - 1 + 1) * N, i * N:(i + 1) * N] = temp

    # 加入自回
    for i in range(len(adj)):
        adj[i, i] = 1
    return adj


class AirDataset(data.Dataset):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample, x_marks,
                 x_meter):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            x_marks_padding = np.repeat(x_marks[-1:], num_padding, axis=0)
            # x_graphs_padding = np.repeat(x_graphs[-1:], num_padding, axis=0)
            x_meter_padding = np.repeat(x_meter[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            x_marks = np.concatenate([x_marks, x_marks_padding], axis=0)
            # x_graphs = np.concatenate([x_graphs, x_graphs_padding], axis=0)
            x_meters = np.concatenate([x_meter, x_meter_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xs_mark = x_marks
        # self.xs_graph = x_graphs
        self.xs_meter = x_meters

    def load_data(self):
        data = []
        target = []
        for i in range(1, 6):
            path = os.path.join(self.root, f'aircraft_{str(i)}.txt')
            data.append(np.loadtxt(path))
            target.append(i)
        return data, target

    def __getitem__(self, index):
        """返回的是一个元组，第一个元组是数据，第二个元组是标签"""
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


"""数据加载器"""


class DataLoader(object):
    def __init__(
            self,
            xs,
            ys,
            batch_size,
            pad_with_last_sample=True,
            x_marks=None,
            #  x_graphs=None,
            x_meter=None):
        """
        数据加载器
        :param xs:训练数据
        :param ys:标签数据
        :param batch_size:batch大小
        :param pad_with_last_sample:剩余数据不够时，是否复制最后的sample以达到batch大小
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            x_marks_padding = np.repeat(x_marks[-1:], num_padding, axis=0)
            # x_graphs_padding = np.repeat(x_graphs[-1:], num_padding, axis=0)
            x_meter_padding = np.repeat(x_meter[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            x_marks = np.concatenate([x_marks, x_marks_padding], axis=0)
            # x_graphs = np.concatenate([x_graphs, x_graphs_padding], axis=0)
            x_meters = np.concatenate([x_meter, x_meter_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xs_mark = x_marks
        # self.xs_graph = x_graphs
        self.xs_meter = x_meters

    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        xs_meter = self.xs_meter[permutation]

        # xs_mark = self.xs_mark[permutation, :]
        B, L, N, D = self.xs_mark.shape[0], self.xs_mark.shape[
            1], self.xs_mark.shape[2], self.xs_mark.shape[3]

        x_mark = np.zeros((B, L, N, D), dtype=self.xs_mark.dtype)
        for i, index in enumerate(permutation):
            x_mark[i, ...] = self.xs_mark[index, ...].copy()
        self.xs = xs
        self.ys = ys
        self.xs_mark = x_mark
        # self.xs_graph = x_graph
        self.xs_meter = xs_meter

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                xmark_i = self.xs_mark[start_ind:end_ind, ...]
                # xgraph_i = self.xs_graph[start_ind:end_ind, ...]
                xmeter_i = self.xs_meter[start_ind:end_ind, ...]

                # yield x_i, y_i, xmark_i, xgraph_i, xmeter_i
                yield x_i, y_i, xmark_i, xmeter_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """标准转换器"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def load_dataset(dataset_dir,
                 normalizer,
                 batch_size,
                 valid_batch_size=None,
                 test_batch_size=None,
                 column_wise=False):
    """
    加载数据集
    :param dataset_dir: 数据集目录
    :param normalizer: 归一方式
    :param batch_size: batch大小
    :param valid_batch_size: 验证集batch大小
    :param test_batch_size: 测试集batch大小
    :param column_wise: 是指列元素的级别上进行归一，否则是全样本取值
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        cat_meterlogical_data = np.load(
            os.path.join(dataset_dir, category + '_meterlogy.npz'))

        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['x_mark_' + category] = cat_data['mark']
        # data['x_graph_'+category] = cat_data['graph']
        data['x_meterlogical_' + category] = cat_meterlogical_data['x']

    if normalizer == 'max01':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
            meter_minimum = data['x_meterlogical_train'].min(axis=0,
                                                             keepdims=True)
            meter_maximum = data['x_meterlogical_train'].max(axis=0,
                                                             keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()
            meter_minimum = data['x_meterlogical_train'].min()
            meter_maximum = data['x_meterlogical_train'].max()

        scaler = MinMax01Scaler(minimum, maximum)
        meter_scaler = MinMax01Scaler(meter_minimum, meter_maximum)
        print('Normalize the dataset by MinMax01 Normalization')

    elif normalizer == 'max11':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
            meter_minimum = data['x_meterlogical_train'].min(axis=0,
                                                             keepdims=True)
            meter_maximum = data['x_meterlogical_train'].max(axis=0,
                                                             keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()
            meter_minimum = data['x_meterlogical_train'].min()
            meter_maximum = data['x_meterlogical_train'].max()

        scaler = MinMax11Scaler(minimum, maximum)
        meter_scaler = MinMax11Scaler(meter_minimum, meter_maximum)
        print('Normalize the dataset by MinMax11 Normalization')

    elif normalizer == 'std':
        if column_wise:
            mean = data['x_train'].mean(axis=0, keepdims=True)  # 获得每列元素的均值、标准差
            std = data['x_train'].std(axis=0, keepdims=True)
            meter_mean = data['x_meterlogical_train'].mean(axis=0,
                                                           keepdim=True)
            meter_std = data['x_meterlogical_train'].std(axis=0, keepdim=True)

        else:
            mean = data['x_train'].mean()
            std = data['x_train'].std()
            meter_mean = data['x_meterlogical_train'].mean()
            meter_std = data['x_meterlogical_train'].std()

        scaler = StandardScaler(mean, std)
        meter_scaler = StandardScaler(meter_mean, meter_std)
        print('Normalize the dataset by Standard Normalization')

    elif normalizer == 'None':
        scaler = NScaler()
        print('Do not normalize the dataset')
    else:
        raise ValueError

    for category in ['train', 'val', 'test']:
        data['x_' + category][...,
        0] = scaler.transform(data['x_' + category][...,
        0])
        data['x_meterlogical_' + category] = meter_scaler.transform(
            data['x_meterlogical_' + category])
    data['train_loader'] = DataLoader(
        data['x_train'],
        data['y_train'],
        batch_size,
        x_marks=data['x_mark_train'],
        #   x_graphs=data['x_graph_train'],
        x_meter=data['x_meterlogical_train'])

    data['val_loader'] = DataLoader(
        data['x_val'],
        data['y_val'],
        valid_batch_size,
        x_marks=data['x_mark_val'],
        # x_graphs=data['x_graph_val'],
        x_meter=data['x_meterlogical_val'])

    data['test_loader'] = DataLoader(
        data['x_test'],
        data['y_test'],
        test_batch_size,
        x_marks=data['x_mark_test'],
        #  x_graphs=data['x_graph_test'],
        x_meter=data['x_meterlogical_test'])
    data['scaler'] = scaler

    return data


"""指标"""


def masked_mse(preds, labels, null_val=np.nan):
    mask = ~torch.isnan(labels) if np.isnan(null_val) else labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    mask = ~torch.isnan(labels) if np.isnan(null_val) else labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# def masked_mape(preds, labels, null_val=np.nan):
#     mask = ~torch.isnan(labels) if np.isnan(null_val) else labels != null_val
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels) / labels
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def masked_r2_score(y_pred, y_true, null_val=np.nan):
#     mask = ~torch.isnan(y_true) if np.isnan(null_val) else y_true != null_val
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#
#     y_true_masked = y_true * mask
#     y_pred_masked = y_pred * mask
#     #flatten
#     y_true_masked = torch.flatten(y_true_masked)
#     y_pred_masked = torch.flatten(y_pred_masked)
#
#     ss_res = torch.sum((y_true_masked - y_pred_masked)**2)
#     ss_tot = torch.sum((y_true_masked - torch.mean(y_true_masked))**2)
#
#     r2_score = 1 - (ss_res/ss_tot)
#     r2_score = torch.where(torch.isnan(r2_score), torch.zeros_like(r2_score), r2_score)
#
#     return torch.mean(r2_score)

def masked_r2_score(y_pred, y_true, null_val=np.nan):
    r2_score_list = []
    y_pred_numpy = y_pred.cpu().detach().numpy().squeeze()
    y_true_numpy = y_true.cpu().detach().numpy().squeeze()
    for i in range(y_pred_numpy.shape[1]):
        r2_score_list.append(r2_score(y_true_numpy[:, i], y_pred_numpy[:, i]))

    return np.mean(r2_score_list)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()

    return mae, mape, rmse


if __name__ == '__main__':
    # adj = get_adjacency_matrix("./data/PEMS04/PEMS04.csv",
    #                            307,
    #                            id_filename=None)
    # print(adj)
    # A = construct_adj(adj, 3)
    # print(A.shape)
    # print(A)

    # dataloader = load_dataset('./data/processed/PEMS04/',
    #                           'std',
    #                           batch_size=64,
    #                           valid_batch_size=64,
    #                           test_batch_size=64)
    # print(dataloader)
    matrix = [[1, 1, 0, 1, 0, 1, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 1],
              [0, 1, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 1, 0],
              [0, 1, 0, 0, 0, 0, 0, 1]]
    mt = np.array(matrix)
    construct_dynamic_adj_lt(mt, 3)
    construct_dynamic_adj_1hop(mt, 3)
