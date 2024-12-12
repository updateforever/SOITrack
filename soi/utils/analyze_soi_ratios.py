import json
import os
import pandas as pd
import numpy as np
import collections


def analyze_retio(dataset, subset):
    data_path = os.path.join('/mnt/second/wangyipei/SOI/data/%s/data' % dataset)
    seq_names = os.listdir(data_path)
    once_soi = []
    more_soi = []
    most_soi = []
    start_frame = 0

    for item in seq_names:
        seq_soi_file = os.path.join(data_path, item, 'soi.txt')
        soi_frames = np.loadtxt(seq_soi_file, delimiter='\n')
        soi_frames = soi_frames.reshape(soi_frames.size)
        soi_num = 0

        # 判断soi分布
        for i, frame in enumerate(soi_frames):
            if frame == 0:
                soi_num = 0
                continue
            if i == 0:
                soi_num += 1
                start_frame = int(frame)
            elif int(frame) - start_frame >= 10:
                soi_num += 1
                start_frame = int(frame)

        '''ratio = soi_num / total_num
        if 0 < ratio <= 0.1:
             once_soi.append(item)
        elif 0.1 < ratio <= 0.3:
            more_soi.append(item)
        elif 0.3 < ratio <= 0.3:
            continue
        else:
            print(soi_num)
            most_soi.append(item)'''
        if soi_num == 1:
            once_soi.append(item)
        elif 1 < soi_num <= 10:
            more_soi.append(item)
        elif soi_num == 0:
            print('error 0 soi_num' + str(item))
        else:
            print(str(soi_num) + 'occur in ' + str(item))
            most_soi.append(item)

    info_path = os.path.join('/mnt/second/wangyipei/SOI/data', dataset)  # subset
    once_soi_info_file = os.path.join(info_path, 'once.json')
    more_soi_info_file = os.path.join(info_path, 'more.json')
    most_soi_info_file = os.path.join(info_path, 'most.json')
    info_json = {}
    with open(once_soi_info_file, 'w') as f:
        info_json[subset] = once_soi
        json.dump(info_json, f, indent=4)

    with open(more_soi_info_file, 'w') as f:
        info_json[subset] = more_soi
        json.dump(info_json, f, indent=4)

    with open(most_soi_info_file, 'w') as f:
        info_json[subset] = most_soi
        json.dump(info_json, f, indent=4)


def analyze_results():
    data_path = os.path.join('/mnt/second/wangyipei/SOI/data/lasot/test/mine_result/data')
    seq_names = os.listdir(data_path)

    data_results = {}
    # 定义一个空的Counter对象
    counter = collections.Counter()
    for item in seq_names:
        seq_soi_file = os.path.join(data_path, item, 'soi.txt')
        soi_frames = np.loadtxt(seq_soi_file, delimiter='\n')
        soi_frames = soi_frames.reshape(soi_frames.size)
        soi_num = 0
        start_frame = 0
        # 判断soi分布
        for i, frame in enumerate(soi_frames):
            if frame == 0:
                soi_num = 0
                continue
            if i == 0:
                soi_num += 1
                start_frame = int(frame)
            elif int(frame) - start_frame >= 10:
                soi_num += 1
                start_frame = int(frame)
        counter.update([soi_num])
    # 打印统计结果
    print(counter)
    # 导入pandas模块
    import pandas as pd

    # 将统计结果转换为DataFrame格式
    df = pd.DataFrame.from_dict(counter, orient="index", columns=["count"])
    df.sort_index(ascending=True,  # ascending=True --升序排序(默认)；  ascending=False --降序排序
                  inplace=True,  # 如果为True,则直接删除，对原df进行操作； 如果为False(默认)，那么返回一个结果，不会对原df操作！
                  axis=0,  # axis=0	--根据行索引排序(默认)；  axis=1	--根据列索引排序
                  key=None)

    # 打印DataFrame
    print(df)

    # 将DataFrame保存为csv文件
    df.to_csv("/mnt/second/wangyipei/SOI/reference/lasotSOI/test/count.csv")


if __name__ == '__main__':
    # analyze_retio('VOTLT2019', 'test')
    analyze_results()
