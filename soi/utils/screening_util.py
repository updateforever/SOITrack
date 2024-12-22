import json
import numpy as np
import torch
import math
import cv2 as cv
import random
import torch.nn.functional as F
from torch import tensor
from .tensor import TensorList
import os



def find_local_maxima_v1(scores, all_boxes, ks=5, s=3, threshold_type='alpha', alpha=0.6, th=0.1, min_s=0.0):
    """
    根据不同的阈值方式在热图中找到局部极大值点。

    参数：
        scores (tensor): 热图数据，形状为 (batch_size, channels, height, width)。
        all_boxes (tensor): 所有候选框信息，形状为 (batch_size, num_points, 4)。
        ks (int): 局部邻域（卷积核大小），定义了两个极大值之间的最小距离。
        threshold_type (str): 阈值类型，'basic' 使用最小得分阈值 `th`，'alpha' 使用最大得分比例阈值 `alpha`。
        alpha (float): 最大得分比例阈值，仅在 `threshold_type='alpha'` 时有效。
        th (float): 最小得分阈值，仅在 `threshold_type='basic'` 时有效。
        min_s (float): 用于平移得分的最小值，避免负数对计算的影响。

    返回：
        tuple: 包含以下三个部分：
            - coords_batch (list or tensor): 每个批次的局部极大值点坐标。
            - intensities_batch (list or tensor): 每个批次局部极大值点的得分。
            - boxes_batch (list or tensor): 每个批次局部极大值点对应的候选框。
    """
    # 1. 平移得分，确保得分非负
    scores = scores - min_s
    max_score = torch.max(scores)  # 最大得分，用于 alpha 阈值计算
    ndims = scores.ndimension()  # 维度判断

    # 2. 确保输入是 4D (batch_size, channels, height, width)
    if ndims == 2:
        scores = scores.view(1, 1, scores.shape[0], scores.shape[1])

    # 3. 使用最大池化查找局部极大值
    scores_max = F.max_pool2d(scores, kernel_size=ks, stride=s, padding=ks // 2)

    # 恢复到原始尺寸
    upsampled_scores_max = F.interpolate(scores_max, size=scores.shape[-2:], mode='nearest')

    # 4. 根据阈值类型筛选局部极大值
    if threshold_type == 'basic':
        peak_mask = (scores == upsampled_scores_max) & (scores >= th)  # 直接根据阈值筛选
    elif threshold_type == 'alpha':
        peak_mask = (scores == upsampled_scores_max) & (scores >= th) & (scores > alpha * max_score)  # 按最大得分比例筛选
    elif threshold_type == 'debug':
        peak_mask = ((scores == upsampled_scores_max) & (scores > alpha * max_score)) | (scores >= th)  # 为了可视化debug
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}. Choose either 'basic' or 'alpha'.")

    # 5. 提取局部极大值的坐标和得分
    coords = torch.nonzero(peak_mask, as_tuple=False)  # 返回局部极大值的坐标 torch.Size([n, 4])
    intensities = scores[peak_mask]  # 提取对应的得分

    # 将 y 和 x 转换为展平后的索引
    y_coords = coords[:, 2]  # 第3个维度表示y坐标
    x_coords = coords[:, 3]  # 第4个维度表示x坐标
    H, W = peak_mask.shape[2], peak_mask.shape[3]  # 特征图高宽 (24, 24)
    flatten_indices = y_coords * W + x_coords  # 计算展平后的索引

    # 6. 按得分从高到低排序
    idx_maxsort = torch.argsort(-intensities)
    coords = coords[idx_maxsort]  # 排序后的坐标  n, 4
    intensities = intensities[idx_maxsort]  # 排序后的得分

    boxes_batch = []

    # 7. 处理批量数据情况
    if ndims == 4:
        coords_batch, intensities_batch = [], []
        for i in range(scores.shape[0]):
            # 筛选属于当前批次的数据
            mask = (coords[:, 0] == i)
            selected_coords = coords[mask, 2:]  # 当前批次的坐标 (height, width)
            selected_intensities = intensities[mask]  # 当前批次的得分

            # 使用选中坐标从 all_boxes 中提取对应的候选框
            selected_boxes = all_boxes[i, :, flatten_indices]

            # 将结果添加到对应列表中
            coords_batch.append(selected_coords)
            intensities_batch.append(selected_intensities)
            boxes_batch.append(selected_boxes)
    else:
        coords_batch = coords[:, 2:]  # 提取坐标 (height, width)
        intensities_batch = intensities  # 提取得分
        for i in range(coords.size(0)):
            coo = coords[i]
            boxes_temp = all_boxes[:, :, coo[2], coo[3]]  # 提取对应的候选框 1,4,n
            boxes_batch.append(boxes_temp)
    # 8. 返回局部极大值的坐标、得分和对应的候选框
    return coords_batch, intensities_batch, boxes_batch


def load_dump_seq_data_from_disk(path):
    """
    从磁盘加载序列数据。
    
    参数：
        path (str): 数据文件路径。

    返回：
        dict: 存储序列数据的字典。
    """
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    return data


def dump_seq_data_to_disk(save_path, seq_name, seq_data):
    """
    将序列数据保存到磁盘。
    
    参数：
        save_path (str): 保存路径。
        seq_name (str): 序列名称。
        seq_data (dict): 序列数据。
    """
    data = load_dump_seq_data_from_disk(save_path)
    data[seq_name] = seq_data
    with open(save_path, 'w') as f:
        json.dump(data, f)


def load_jsonl_file(path):
    """
    从 JSONL 文件中加载数据。
    
    参数：
        path (str): JSONL 文件路径。
    
    返回：
        list: 包含所有数据的列表。
    """
    data = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
    return data


def convert_to_serializable(obj):
    """
    将对象中的 Tensor 转换为 JSON 可序列化的类型。
    """
    if isinstance(obj, torch.Tensor):  # 如果是 Tensor，转换为列表或标量
        return obj.tolist() if obj.dim() > 0 else obj.item()
    elif isinstance(obj, dict):  # 如果是字典，递归转换每个值
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # 如果是列表，递归转换每个元素
        return [convert_to_serializable(v) for v in obj]
    else:  # 其他类型直接返回
        return obj

def append_to_jsonl_file(path, seq_name, seq_data):
    """
    向 JSONL 文件追加一条数据。
    
    参数：
        path (str): JSONL 文件路径。
        seq_name (str): 序列名称。
        seq_data (dict): 序列数据。
    """
    # 将 seq_data 转换为 JSON 可序列化的格式
    serializable_seq_data = convert_to_serializable(seq_data)
    
    # 组织记录并写入 JSONL 文件
    record = {"seq_name": seq_name, "seq_data": serializable_seq_data}
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')
        

def determine_frame_state(tracking_data, tracker, seq, th=0.25):
    """
    判断当前帧的状态。
    
    参数：
        tracking_data (dict): 跟踪数据。
        tracker: 跟踪器对象。
        seq: 序列对象。
        th (float): 阈值，用于判断候选分数。

    返回：
        str: 帧的状态（如 'G', 'H', 'J', 'K'），如果无法判断返回 None。
    """
    visible = seq.target_visible[tracker.frame_num - 1]
    num_candidates = tracking_data['target_candidate_scores'].shape[0]
    state = None

    if num_candidates >= 2:
        max_candidate_score = tracking_data['target_candidate_scores'].max()

        # 计算候选框与目标框的距离
        anno_and_target_candidate_score_dists = torch.sqrt(
            torch.sum((tracking_data['target_anno_coord'] - tracking_data['target_candidate_coords']) ** 2,
                      dim=1).float())
        ids = torch.argsort(anno_and_target_candidate_score_dists)

        score_dist_pred_anno = anno_and_target_candidate_score_dists[ids[0]]
        sortindex_correct_candidate = ids[0]
        score_dist_anno_2nd_highest_score_candidate = anno_and_target_candidate_score_dists[ids[1]] if num_candidates > 1 else 10000

        # 根据得分和距离判断状态
        if (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
                sortindex_correct_candidate == 0 and max_candidate_score < th and visible != 0):
            state = 'G'
        elif (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
              sortindex_correct_candidate == 0 and max_candidate_score >= th and visible != 0):
            state = 'H'
        elif (num_candidates > 1 and score_dist_pred_anno > 4 and max_candidate_score >= th and visible != 0):
            state = 'J'
        elif (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
              sortindex_correct_candidate > 0 and max_candidate_score >= th and visible != 0):
            state = 'K'

    return state


def determine_subseq_state(frame_state, frame_state_previous):
    """
    判断子序列的状态。
    
    参数：
        frame_state (str): 当前帧的状态。
        frame_state_previous (str): 上一帧的状态。

    返回：
        str: 子序列状态（如 'GH', 'HK'），如果无法判断返回 None。
    """
    if frame_state is not None and frame_state_previous is not None:
        return '{}{}'.format(frame_state_previous, frame_state)
    else:
        return None


def extract_candidate_data(data, th=0.1):
    """
    提取候选数据。
    
    参数：
        data (dict): 跟踪数据字典。
        th (float): 阈值，用于选择候选分数。

    返回：
        tuple: 包含候选框数据的字典和候选数量。
    """
    search_area_box = data['search_area_box']
    score_map = data['score_map'].cpu()
    all_boxes = data['all_scoremap_boxes']

    target_candidate_coords, target_candidate_scores, candidate_boxes = find_local_maxima_v1(score_map.squeeze(), all_boxes, th=th, ks=5, threshold_type='alpha')

    tg_num = len(target_candidate_scores)

    if 'x_dict' in data.keys():
        search_img = torch.tensor(data['x_dict'])  # data['x_dict'].tensors
        return dict(search_area_box=search_area_box,
                    target_candidate_scores=target_candidate_scores,
                    target_candidate_coords=target_candidate_coords,
                    tg_num=tg_num, 
                    search_img=search_img,
                    candidate_boxes=candidate_boxes)
    return dict(search_area_box=search_area_box,
                target_candidate_scores=target_candidate_scores,
                target_candidate_coords=target_candidate_coords,
                tg_num=tg_num,
                candidate_boxes=candidate_boxes
                )


def update_seq_data(seq_candidate_data, frame_candidate_data):
    """
    更新序列数据，将帧数据添加到序列数据中。
    
    参数：
        seq_candidate_data (dict): 序列候选数据，包含多个帧的候选信息。
        frame_candidate_data (dict): 当前帧的候选数据。
    """
    # 遍历每个键值对
    for key, val in frame_candidate_data.items():
        # 如果值是Tensor类型，将其转为float并转换为列表
        val = val.float().tolist() if torch.is_tensor(val) else val
        # 将当前帧的值添加到序列数据中
        seq_candidate_data[key].append(val)
