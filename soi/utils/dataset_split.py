import os
import random
import shutil

# 源数据集路径和目标数据集路径
path_source = '/mnt/second/hushiyu/UAV/BioDrone/data/train'
path_txt = '/home/micros/projects/pytracking-master/ltr/data_specs/BioDrone_train_train_split.txt'


# 参数：源路径、目标路径和测试集所占比例
def seperate(path_source, path_target, percent):
    # 生成包含path_source下所有目录名的列表
    categories = os.listdir(path_source)
    for name in categories:
        # 在path_target下建立相同名称的子目录
        os.makedirs(os.path.join(path_target, name))
        # 生成包含子目录下所有图片的列表
        nums = os.listdir(os.path.join(path_source, name))
        # 随机按比例抽取一部分图片
        nums_target = random.sample(nums, int(len(nums) * percent))
        # 把图片剪切到目标路径
        for pic in nums_target:
            shutil.move(os.path.join(path_source, name, pic), os.path.join(path_target, name, pic))

# 执行完成后，path_source为训练集，path_target为测试集。
