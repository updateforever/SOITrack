import torch
from torchvision.ops.boxes import box_area
import numpy as np
import math

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)

def box_xywh_to_cxcywh(x):
    x1, y1, w, h = x.unbind(-1)
    b = [(x1 + 0.5 * w), (y1 + 0.5 * h), w, h]
    return torch.stack(b, dim=-1)

def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def generalized_box_iou_V2(boxes1, boxes2, iou_type):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # (N,2)
    
    if iou_type == 'giou':
        area = wh[:, 0] * wh[:, 1] # (N,)
        giou = iou - (area - union) / area
        return giou, iou
    
    elif iou_type == 'siou':
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (boxes2[:, 0] + boxes2[:, 2] - boxes1[:, 0] - boxes1[:, 2]) * 0.5
        s_ch = (boxes2[:, 1] + boxes2[:, 3] - boxes1[:, 1] - boxes1[:, 3]) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / wh[:, 0]) ** 2
        rho_y = (s_ch / wh[:, 1]) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
        w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        siou = iou - 0.5 * (distance_cost + shape_cost)
        return siou , iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]


# def clip_box_batch(boxes: list, H, W, margin=0):
#     clipped_boxes = []
#     for box in boxes:
#         x1, y1, w, h = box
#         x2, y2 = x1 + w, y1 + h
        
#         # Clip the box coordinates to ensure it is within the image boundaries with margin consideration
#         x1 = min(max(0, x1), W-margin)
#         x2 = min(max(margin, x2), W)
#         y1 = min(max(0, y1), H-margin)
#         y2 = min(max(margin, y2), H)
        
#         # Ensure width and height are not smaller than the margin
#         w = max(margin, x2 - x1)
#         h = max(margin, y2 - y1)
        
#         # Append the clipped box to the result list
#         clipped_boxes.append([x1, y1, w, h])
    
#     return clipped_boxes


def clip_box_batch(boxes: torch.Tensor, H: int, W: int, margin: int = 0):
    """
    输入:
        boxes: (1, 4, 24, 24) 格式的框 (x1, y1, w, h)
        H: 图像高度
        W: 图像宽度
        margin: 边界余量
    输出:
        clipped_boxes: (1, 4, 24, 24)，裁剪后的框
    """
    # 解析输入的坐标 (x1, y1, w, h)
    x1 = boxes[:, 0, :, :]  # x1
    y1 = boxes[:, 1, :, :]  # y1
    w = boxes[:, 2, :, :]   # 宽度
    h = boxes[:, 3, :, :]   # 高度

    # 计算 x2 和 y2
    x2 = x1 + w
    y2 = y1 + h

    # 裁剪框的坐标，确保在图像边界内
    x1_clipped = torch.clamp(x1, min=0, max=W - margin)
    x2_clipped = torch.clamp(x2, min=margin, max=W)
    y1_clipped = torch.clamp(y1, min=0, max=H - margin)
    y2_clipped = torch.clamp(y2, min=margin, max=H)

    # 重新计算裁剪后的宽度和高度，确保不小于 margin
    w_clipped = torch.clamp(x2_clipped - x1_clipped, min=margin)
    h_clipped = torch.clamp(y2_clipped - y1_clipped, min=margin)

    # 拼接成 (x1, y1, w, h) 格式
    clipped_boxes = torch.stack([x1_clipped, y1_clipped, w_clipped, h_clipped], dim=1)

    return clipped_boxes



def iouhead_loss(src_iouh, iou):
    # For IoU Head: L2 Loss
    loss = torch.mean(((1-iou)**2)*((src_iouh - iou)**2))
    return loss

def siou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    siou, iou = generalized_box_iou_V2(boxes1, boxes2)
    return (1 - siou).mean(), iou