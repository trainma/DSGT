import json

if __name__ == "__main__":
    with open('./data/TianJingAir/TianJIng_station_distrcit.json', 'r') as f:
        tianjin_json = json.load(f)
    print(tianjin_json)
    dict = {}
    # 根据district 划分
    for station in tianjin_json:
        station_dict = tianjin_json[station]
        if dict.get(station_dict['district']) is None:
            dict[station_dict['district']] = []
        dict[station_dict['district']].append(station_dict['station_id'])
    print(dict)
    # dict 保存为json
    data = json.dumps(dict, indent=4,ensure_ascii=False)
    with open('./data/TianJingAir/TianJIng_district_station.json', 'w') as f:
        f.write(data)

    # df = pd.DataFrame(dict)
    # df.to_csv('./data/TianJingAir/TianJing_district_station.csv')
