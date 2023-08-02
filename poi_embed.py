import numpy as np
import json
import pandas as pd
from utils import StandardScaler
import matplotlib.pyplot as plt


def poi_convert(pois_json='./data/BeiJingAir/Station_pois.json',
                pois_txt='./data/BeiJingAir/Station_pois.txt'):
    with open(pois_json, encoding='utf-8') as a:
        pois = json.load(a)
    with open(pois_txt, 'w', encoding='gbk') as fp:
        for station in pois:
            for category in pois[station]:
                print(station + '-' + category + '-' + str(pois[station][category]))
                fp.write(station + '-' + category + '-' + str(pois[station][category]) + '\n')


if __name__ == '__main__':
    # poi_convert('./data/TianJingAir/TianJing_Station_pois.json', './data/TianJingAir/Station_pois.txt')
    # print(pois)
    df = pd.read_csv('./data/TianJingAir/Station_pois.csv',
                     sep=' ', header=None)
    mean = df.iloc[:, 2].mean()
    std = df.iloc[:, 2].std()
    scaler = StandardScaler(mean, std)
    scaler_data = scaler.transform(df.iloc[:, 2].values)
    df.iloc[:, 2] = scaler_data

    df.head()
    df.to_csv('./data/TianJingAir/Station_pois_scaler.csv', index=False)
