import json
import math
import os

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler


def compute_gra(data, time_window):
    assert time_window == data.shape[0]

    delta_matrix = np.zeros((time_window, data.shape[1], data.shape[1]))

    # 计算灰色关联度的delta矩阵

    for step in range(data.shape[0]):
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if j == i:
                    continue
                delta_matrix[step, i, j] = abs(data[step, i] - data[step, j])

    # 计算灰色关联度的相关性矩阵
    roi = 0.5
    # compute the max of delta matrix
    delta_max = np.max(np.max(delta_matrix, axis=0), axis=1)
    # compute the min of delta matrix
    delta_min = np.min(np.min(delta_matrix, axis=0), axis=1)
    corr_matrix = np.zeros((time_window, data.shape[1], data.shape[1]))

    for step in range(data.shape[0]):
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if j == i:
                    continue
                A = delta_min[i] + roi * delta_max[i]
                B = delta_matrix[step, i, j] + roi * delta_max[i]
                corr_matrix[step, i, j] = A / B

    R = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            R[i, j] = np.sum(corr_matrix[:, i, j]) / len(corr_matrix)

    for i in range(data.shape[1]):
        R[i, i] = 1

    return R


def rbf_kernel(dist, sigma2):  # RBF kernel function
    return np.exp(-1 * ((dist ** 2) / (sigma2)))


def compute_spatial_graph(path, loc_list, threshold=0.1, sigma=1000):
    with open(path, 'r', encoding='utf-8') as f:
        station_dict = json.load(f)
    loc_list = ['万寿西宫', '前门', '亦庄', '定陵', '西直门北', '官园', '门头沟', '永乐店', '怀柔', '万柳', '东四环',
                '农展馆', '奥体中心', '永定门内', '琉璃河', '云岗', '八达岭', '密云水库', '天坛', '房山', '榆垡',
                '顺义', '大兴', '昌平', '古城', '丰台花园', '通州', '东四', '南三环', '密云', '平谷', '北部新区',
                '东高村', '延庆']

    spatial_graph = np.zeros((len(loc_list), len(loc_list)))
    for i in range(len(loc_list)):
        for j in range(len(loc_list)):
            if i == j:
                spatial_graph[i, j] = 1
            else:
                spatial_distance = geodesic(
                    (station_dict[str(loc_list[i])]['纬度'],
                     station_dict[str(loc_list[i])]['经度']),
                    (station_dict[str(loc_list[j])]['纬度'], station_dict[str(loc_list[j])]['经度'])).km
                W_ij = rbf_kernel(spatial_distance, sigma)

                spatial_graph[i, j] = 0 if W_ij < threshold else W_ij
    return spatial_graph


def compute_spatial_graph2(path, loc_list, threshold=0.1, sigma=1000):
    with open(path, 'r', encoding='utf-8') as f:
        station_dict = json.load(f)

    loc_dict = {loc: (station_dict[loc]['纬度'], station_dict[loc]['经度']) for loc in loc_list}

    spatial_graph = np.zeros((len(loc_list), len(loc_list)))

    # Iterate over the upper triangular matrix
    for i in range(len(loc_list)):
        for j in range(i + 1, len(loc_list)):
            spatial_distance = geodesic(loc_dict[loc_list[i]], loc_dict[loc_list[j]]).km
            W_ij = rbf_kernel(spatial_distance, sigma)
            spatial_graph[i, j] = W_ij
            spatial_graph[j, i] = W_ij  # Assign to the lower triangular matrix too

    np.fill_diagonal(spatial_graph, 1)  # Set diagonal values to 1

    spatial_graph[spatial_graph < threshold] = 0  # Thresholding

    return spatial_graph


def compute_pois_graph():
    with open(os.path.join('./BeiJingAir/', 'Station_pois.json'), 'r') as f:
        station_dict = json.load(f)
    loc_list = ['万寿西宫', '前门', '亦庄', '定陵', '西直门北', '官园', '门头沟', '永乐店', '怀柔', '万柳', '东四环',
                '农展馆', '奥体中心', '永定门内', '琉璃河', '云岗', '八达岭', '密云水库', '天坛', '房山', '榆垡',
                '顺义', '大兴', '昌平', '古城', '丰台花园', '通州', '东四', '南三环', '密云', '平谷', '北部新区',
                '东高村', '延庆']

    station_list = []
    for loc in loc_list:
        station_data = station_dict[loc]
        temp_list = [station_data[cls] for cls in station_data.keys()]
        station_list.append(temp_list)

    st_matrix = np.array(station_list)
    poi_corr = cdist(st_matrix, st_matrix, 'euclidean')
    poi_corr = poi_corr / np.max(poi_corr)
    for i in range(poi_corr.shape[0]):
        poi_corr[i, i] = 1  # add self-loop
    return poi_corr


def compute_corr_graph():
    with open(os.path.join('./BeiJingAir/', 'Station_pois.json'), 'r') as f:
        station_dict = json.load(f)
    loc_list = ['万寿西宫', '前门', '亦庄', '定陵', '西直门北', '官园', '门头沟', '永乐店', '怀柔', '万柳', '东四环',
                '农展馆', '奥体中心', '永定门内', '琉璃河', '云岗',
                '八达岭', '密云水库', '天坛', '房山', '榆垡', '顺义', '大兴', '昌平', '古城', '丰台花园', '通州',
                '东四', '南三环', '密云', '平谷', '北部新区', '东高村', '延庆']

    station_list = []
    for loc in loc_list:
        station_data = station_dict[loc]
        temp_list = [station_data[cls] for cls in station_data.keys()]
        station_list.append(temp_list)
    st_matrix = np.array(station_list)
    # st_matrix = st_matrix.T
    poi_corr = cdist(st_matrix, st_matrix, 'euclidean')
    poi_corr = poi_corr / np.max(poi_corr)
    for i in range(poi_corr.shape[0]):
        poi_corr[i, i] = 1  # add self-loop
    return poi_corr


def main() -> None:
    loc_list = ['万寿西宫', '前门', '亦庄', '定陵', '西直门北', '官园', '门头沟', '永乐店', '怀柔', '万柳', '东四环',
                '农展馆', '奥体中心', '永定门内', '琉璃河', '云岗', '八达岭', '密云水库', '天坛', '房山', '榆垡',
                '顺义', '大兴', '昌平', '古城', '丰台花园', '通州', '东四', '南三环', '密云', '平谷', '北部新区',
                '东高村', '延庆']
    spatial_graph = compute_spatial_graph('./data/BeiJingAir/Station.json', loc_list, 0.3)
    print(spatial_graph)
    # spatial_graph_df = pd.DataFrame(spatial_graph)
    # spatial_graph_df.to_csv('./data/BeiJingAir/spatial_corr2.csv')
    # AirData = np.load('./BeiJingAir/Air.npz')['data']

    # R = compute_GRA(AirData,12)
    # with open('./station.json', 'r') as f:
    #     station = json.load(f)
    # df = pd.read_csv('./PM2_5.csv', index_col=0)
    # S = np.zeros((len(df.columns), len(df.columns)))

    # data = train_data[0,...]
    # data = data.squeeze(-1)
    # R = compute_GRA(data,12)

    # graph = R * spatial_graph
    # print(spatial_graph)
    # poi_corr = compute_pois_graph()
    # spatial_corr = compute_spatial_graph()

    # train_data = np.load('./processed/Air/train.npz')['x']
    # for i in range(train_data.shape[0]):
    #     data = train_data[i, ...]
    #     data = data.squeeze(-1)
    #     GRA_graph = compute_GRA(data, 12)
    #     graph = GRA_graph * spatial_corr * poi_corr
    # path = './data/TianJingAir/TianJIng_station_dict.json'
    # loc_list = [6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6010, 6011, 6012, 6013, 6014, 6015, 6016, 6017, 6019,
    #             6020, 6021, 6022, 6023, 6024, 6025, 6026, 6027, 6028, 6040]
    # spatial_graph = compute_spatial_graph(path, loc_list)
    # spatial_graph_df = pd.DataFrame(spatial_graph)
    # spatial_graph_df.to_csv('./data/TianJingAir/spatial_graph.csv')
    # print(spatial_graph)
    # print("Have Done！")


if __name__ == '__main__':
    main()
