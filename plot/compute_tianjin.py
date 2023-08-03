from utils import *
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

if __name__ == "__main__":
    with open('./data/TianJingAir/TianJIng_district_station.json', 'r') as f:
        tianjin_json = json.load(f)
    temp_df = pd.read_csv('./data/TianJingAir/3sigma.csv')
    tianjin_pred = np.load('./yhat_1_Tianjin.npy')
    tianjin_real = np.load('./yreal_1_Tianjin.npy')
    tianjin_pred = tianjin_pred.squeeze((1, -1))
    tianjin_real = tianjin_real.squeeze((1, -1))
    real = pd.DataFrame(tianjin_real, columns=list(temp_df.columns[1:]))
    pred = pd.DataFrame(tianjin_pred, columns=list(temp_df.columns[1:]))
    df_real = pd.DataFrame(np.zeros((tianjin_real.shape[0], len(tianjin_json))), columns=list(tianjin_json.keys()))
    df_pred = pd.DataFrame(np.zeros((tianjin_real.shape[0], len(tianjin_json))), columns=list(tianjin_json.keys()))

    for district in tianjin_json:
        print(district)
        station_list = tianjin_json[district]
        station_list = [str(i) for i in station_list]
        print(station_list)
        pred_temp = pred[station_list].values
        pred_temp = pred_temp.reshape(-1, len(station_list))
        pred_temp = np.mean(pred_temp, axis=-1)
        real_temp = real[station_list].values
        real_temp = real_temp.reshape(-1, len(station_list))
        real_temp = np.mean(real_temp, axis=-1)
        df_real[district] = real_temp
        df_pred[district] = pred_temp
    df_real.to_excel('tianjin1_district.xlsx')
    df_pred.to_excel('tianjin1_district_pred.xlsx')

