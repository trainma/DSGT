# %%
import numpy as np
import torch
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
real = np.load('./yreal.npy')
predict = np.load('./yhat.npy')
# %%
from utils import *

mse = masked_mse(torch.from_numpy(predict), torch.from_numpy(real), 0.0).item()
mae = masked_mae(torch.from_numpy(predict), torch.from_numpy(real), 0.0).item()
rmse = masked_rmse(torch.from_numpy(predict), torch.from_numpy(real), 0.0).item()
print('mae: ', mae)
print('rmse: ', rmse)
# %%
mae_station = []
rmse_station = []
for i in range(predict.shape[2]):
    mae_station.append(masked_mae(
        torch.from_numpy(predict[:, :, i, :]), torch.from_numpy(real[:, :, i, :])))
    rmse_station.append(masked_rmse(torch.from_numpy(predict[:, :, i, :])
                                    , torch.from_numpy(real[:, :, i, :])))
mae_station = np.array(mae_station)
rmse_station = np.array(rmse_station)
# %%
DongCheng = [0, 1, 30]
HaiDing = [2, 6, 7, 31]
FengTai = [3, 8, 9, 32]
ChaoYang = [4, 5, 33]
ShiJingShan = [10]
Fangshan = [11, 28]
DaXing = [12, 13, 27]
Tongzhou = [14, 26]
ShunYi = [15]
ChangPing = [16, 22]
MengTouGou = [17]
PingGu = [18, 25]
HuaiRou = [19]
MiYun = [20, 24]
YanQing = [21, 23]
XiCheng = [29]
District = [DongCheng, XiCheng, ChaoYang, FengTai, ShiJingShan, HaiDing, MengTouGou, Fangshan,
            Tongzhou, ShunYi, ChangPing, DaXing, HuaiRou, PingGu, MiYun, YanQing]

# %%
# 按District数组分别计算不同区域站点的平均误差
District_mae = []
District_rmse = []

for i in range(len(District)):
    print('District: ', District[i])
    District_mae.append(np.mean(mae_station[District[i]]))
    District_rmse.append(np.mean(rmse_station[District[i]]))
    print('mae: ', np.mean(mae_station[District[i]]))
    print('rmse: ', np.mean(rmse_station[District[i]]))

# %%
dis_mae = np.array(District_mae)
dis_rmse = np.array(District_rmse)
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'  # 设置字体为微软雅黑
# 使打印输出显示更全
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# 读取北京geojson数据
data = gpd.read_file('./data/BeiJingAir/BJ.json')
print(data)

# %%
BJ_district_plot = pd.concat(
    [data, pd.DataFrame({'mae': dis_mae, 'rmse': dis_rmse})], axis=1)
BJ_district_plot.head()
# %%
fig, ax = plt.subplots(figsize=(12, 9), dpi=500)
p = BJ_district_plot.plot(column='mae', scheme='BoxPlot', ax=ax, ec='black', lw=0.7, legend=True,
              legend_kwds={'loc': 'center left',
                           'bbox_to_anchor': (1, 0.5), 'interval': True}, cmap='autumn')
print(p)

