import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        print('mean:', self.mean, 'std:', self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class Air_dataset(Dataset):
    def __init__(self,
                 data='BeiJing',
                 flag='train',
                 seq_len=12,
                 pred_len=12):
        if data == 'Beijing':
            self.root_path = './data/BeiJingAir/'
            self.data_path = 'BeijingAir.csv'
            self.meter_path = './data/BeiJingAir/Beijing_Metrology.csv'
            self.TimeIndex = pd.date_range(
                "2018-01-02", "2021-01-01", freq="1H")[:-1]
        elif data == 'Tianjin':
            self.root_path = './data/TianJingAir/'
            self.data_path = '3sigma.csv'
            self.meter_path = './data/TianJingAir/TianjinMeterology.csv'
            self.TimeIndex = pd.date_range(
                "2014-05-01", "2015-05-01", freq="1H")[:-1]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag
        self.dataset = data
        self.__read_data__()

    def __read_data__(self):

        global num_train, num_val, num_test
        TimeIndex = self.TimeIndex
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path),
                             index_col='datetime', encoding='gbk')
        meterology_raw = pd.read_csv(self.meter_path, index_col=0)

        df_raw.set_index(TimeIndex, inplace=True)
        df_rec = df_raw.copy()

        num_samples = len(df_raw)  # -self.seq_len-self.pred_len+1
        if self.dataset == 'Tianjin':
            num_train = round(num_samples * 0.8)
            num_test = round(num_samples * 0.1)
            num_val = num_samples - num_train - num_test
        elif self.dataset == 'Beijing':
            num_train = round(num_samples * 0.7)
            num_test = round(num_samples * 0.2)
            num_val = num_samples - num_train - num_test

        train_data = df_raw.values[:num_train]
        train_meter = meterology_raw.values[:num_train]
        self.scaler = StandardScaler(train_data.mean(), train_data.std())
        self.meter_scaler = StandardScaler(train_meter.mean(),
                                           train_meter.std())
        borders = {
            'train': [0, num_train],
            'val': [num_train, num_train + num_val],
            'test': [num_samples - num_test, num_samples]
        }

        border1, border2 = borders[self.flag][0], borders[self.flag][1]

        data_raw = df_raw.values[border1:border2]
        data_meter = meterology_raw.values[border1:border2]
        sclaer_data_meter = self.meter_scaler.transform(data_meter)

        data_rec = df_rec.values[border1:border2]
        data = self.scaler.transform(data_raw)
        Time = df_raw.index
        dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
        timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                    // 3600
        timeofday = np.reshape(timeofday, newshape=(-1, 1))
        Time = np.concatenate((dayofweek, timeofday), axis=-1)

        self.data_x = data
        self.data_meter = sclaer_data_meter
        self.data_y = data_rec
        self.data_stamp = Time[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        t_begin = r_begin
        t_end = r_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_meter = self.data_meter[s_begin:s_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[t_begin:t_end]

        nodes = seq_x.shape[-1]
        seq_x = np.expand_dims(seq_x, -1)
        # seq_meter = np.expand_dims(seq_meter, -1)
        seq_y = np.expand_dims(seq_y, -1)

        seq_x_mark = np.tile(np.expand_dims(seq_x_mark, -2),
                             [1, nodes, 1])  # (12,2) -> (12,1,2) -> (12,207,2)
        seq_y_mark = np.tile(np.expand_dims(seq_y_mark, -2), [1, nodes, 1])
        # seq_y_mark按照seq_x_mark 的shape进行扩展
        if seq_x_mark.shape[0] != seq_y_mark.shape[0]:
            length = seq_x_mark.shape[0] - seq_y_mark.shape[0]
            hour = seq_y_mark[-1, 0, 1]
            day = seq_y_mark[-1, 0, 0]
            nodes = seq_y_mark.shape[1]
            for _ in range(length):
                hour = hour + 1
                if hour == 24:
                    hour = 0
                    day = day + 1
                    if day == 7:
                        day = 0
                need_concat = np.array([[day, hour]]).reshape(1, 1, 2)
                need_concat = np.tile(need_concat, [1, nodes, 1])
                seq_y_mark = np.concatenate((seq_y_mark, need_concat), axis=0)

        # if seq_x_mark.shape != seq_y_mark.shape:
        #     length = seq_y_mark.shape[1]-seq_x_mark.shape[1]
        #     end = seq_y_mark[-1]
        #
        #     for i in range(length):

        assert seq_y_mark.shape == seq_x_mark.shape
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_meter

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def read_poi_embedding(poi_path):
    with open(poi_path, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        POIE = torch.zeros((num_vertex, dims), dtype=torch.float32)  # SE p
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            POIE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    return POIE


if __name__ == '__main__':
    dataset = Air_dataset()
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    print(dataset[0][3].shape)
