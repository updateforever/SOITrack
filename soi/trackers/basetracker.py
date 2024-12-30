from __future__ import absolute_import

from typing import Union

import PIL.Image
import torch

import numpy as np
import time

import cv2 as cv

# from ..utils.metrics import iou
import concurrent.futures
from collections import defaultdict
from soi.utils.screening_util import *
from ..utils.help import makedir
from ..utils.metrics import iou


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
        if self.is_using_cuda:
            print("Detect the CUDA devide")
            self._timer_start = torch.cuda.Event(enable_timing=True)
            self._timer_stop = torch.cuda.Event(enable_timing=True)
        self._timestamp = None

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    @property
    def is_using_cuda(self):
        self.cuda_num = torch.cuda.device_count()
        if self.cuda_num == 0:
            return False
        else:
            return True

    def _start_timing(self) -> Union[float, None]:
        if self.is_using_cuda:
            self._timer_start.record()
            timestamp = None
        else:
            timestamp = time.time()
            self._timestamp = timestamp
        return timestamp

    def _stop_timing(self) -> float:
        if self.is_using_cuda:
            self._timer_stop.record()
            torch.cuda.synchronize()
            # cuda event record return duration in milliseconds.
            duration = self._timer_start.elapsed_time(self._timer_stop)
            duration /= 1000.0
        else:
            duration = time.time() - self._timestamp
        return duration

    def track(
        self,
        seq_name,
        img_files,
        anno,
        visualize=False,
        seq_result_dir=None,
        save_img=False,
        mask_info=None,
    ):
        frame_num = len(img_files)
        box = anno[0, :]  # the information of the first frame
        boxes = np.zeros((frame_num, 4))  # save the tracking result
        boxes[0] = box
        times = np.zeros(frame_num)  # save time

        init_positions = []  # save the restart locations
        if visualize:
            display_name = "Display: " + seq_name
            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 960, 720)

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     executor.map(cv.imread, img_files)

        for f, img_file in enumerate(img_files):

            image = cv.imread(img_file)
            height = image.shape[0]
            width = image.shape[1]
            img_resolution = (width, height)

            # start_time = time.time()
            self._start_timing()
            if f == 0:
                self.init(image, box)
                times[f] = self._stop_timing()
            else:
                # mask run
                if mask_info:
                    mask_boxes, gt_coord = find_mask_info_for_frame(
                        mask_info,
                        seq_name,
                        f,
                    )

                    if mask_boxes is not None:  # 如果找到
                        # 进行遮挡 + 恢复 GT
                        image = mask_image_with_boxes(
                            image,
                            mask_boxes,
                            gt_coord,
                            debug_save_path="{}/{:>06d}.jpg".format(seq_result_dir, f),
                        )

                frame_box = self.update(image)
                frame_box = np.rint(frame_box)
                times[f] = self._stop_timing()

                current_gt = anno[f, :].reshape((1, 4))
                frame_box = np.array(frame_box)
                track_result = frame_box.reshape((1, 4))

                boxes[f, :] = frame_box
                # print(seq_name, self.name, ' Tracking %d/%d' % (f, frame_num - 1), 'time:%.2f' % times[f], frame_box)

                if save_img or visualize:
                    frame_disp = image.copy()
                    state = [int(s) for s in frame_box]
                    state[0] = 0 if state[0] < 0 else state[0]
                    state[1] = 0 if state[1] < 0 else state[1]
                    state[2] = (width - state[0] if state[0] + state[2] > width else state[2])
                    state[3] = (height - state[1] if state[1] + state[3] > height else state[3])
                    font_face = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(
                        frame_disp,
                        "No.%06d" % (f),
                        (50, 100),
                        font_face,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    if (anno[f, :] != np.array([0, 0, 0, 0])).all():
                        cv.putText(
                            frame_disp,
                            "seq iou: %2f" % (seq_iou),
                            (50, 130),
                            font_face,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                    cv.rectangle(
                        frame_disp,
                        (state[0], state[1]),
                        (state[2] + state[0], state[3] + state[1]),
                        (0, 255, 0),
                        5,
                    )
                    gt = [int(s) for s in anno[f, :]]
                    cv.rectangle(
                        frame_disp,
                        (gt[0], gt[1]),
                        (gt[2] + gt[0], gt[3] + gt[1]),
                        (0, 0, 255),
                        5,
                    )

                if visualize:
                    cv.imshow(display_name, frame_disp)
                if save_img:
                    if f % 2 == 0:  # 隔一张保存
                        save_path = "{}/{:>06d}.jpg".format(seq_result_dir, f)
                        cv.imwrite(save_path, frame_disp)
                # key = cv.waitKey(1)
                # if key == ord('q'):
                #     break

        if visualize:
            cv.destroyAllWindows()

        return boxes, times

    def track_and_filter_candidates(
        self,
        img_files,
        img_path,
        anno,
        seq_name,
        threshold=0.1,
        track_vis=False,
        heatmap_vis=False,
        masked=True,
        save_masked_img=False,
    ):
        """
        根据目标框在视频序列中进行追踪，并筛选出包含目标候选物的帧数据。

        参数：
            img_files (list): 图像文件路径列表。
            anno (numpy.ndarray): 第一个帧的目标框信息。
            threshold (float): 筛选候选物体的得分阈值，默认值为0.1。

        返回：
            seq_candidate_data (defaultdict): 包含每帧候选数据的字典。
            scores (numpy.ndarray): 每帧的候选数目（得分）。
        """
        frame_num = len(img_files)
        box = anno[0, :]  # 获取第一帧的目标框信息
        boxes = np.zeros((frame_num, 4))  # save the tracking result
        boxes[0] = box
        scores = np.zeros((frame_num, 1), dtype=int)  # 保存每帧的追踪结果（候选物体数量）
        scores[0] = 1  # 第一帧的初始得分为1
        seq_candidate_data = defaultdict(list)  # 用于存储所有帧的候选数据
        display_name = "Display: " + seq_name

        for f, img_file in enumerate(img_files):
            image = cv.imread(img_file)
            height = image.shape[0]
            width = image.shape[1]
            img_resolution = (width, height)

            if f == 0:
                self.init(image, box)  # 初始化追踪器，第一帧
            else:
                frame_box = self.update(image)  # 更新追踪器，进行追踪

                # 获取当前帧的候选数据和候选物体数量
                frame_candidate_data = extract_candidate_data(self.tracker.distractor_dataset_data, th=threshold)

                # 更新序列数据，加入当前帧的候选数据
                update_seq_data(seq_candidate_data, frame_candidate_data)

                frame_box = np.rint(frame_box)

                current_gt = anno[f, :].reshape((1, 4))
                frame_box = np.array(frame_box)
                track_result = frame_box.reshape((1, 4))
                bound = img_resolution
                seq_iou = iou(current_gt, track_result, bound=bound)

                boxes[f, :] = frame_box

                # print(
                #     seq_name,
                #     self.name,
                #     " Tracking %d/%d" % (f, frame_num - 1),
                #     frame_box,
                # )
                # 更新当前帧的候选物体数量（得分）
                scores[f] = frame_candidate_data["tg_num"]

                if track_vis:
                    frame_disp = image.copy()

                    # text
                    font_face = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(
                        frame_disp,
                        "No.%06d" % (f),
                        (0, 20),
                        font_face,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    if (anno[f, :] != np.array([0, 0, 0, 0])).all():
                        cv.putText(
                            frame_disp,
                            "seq iou: %2f" % (seq_iou),
                            (0, 50),
                            font_face,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                    # predict box
                    state = [int(s) for s in frame_box]
                    state[0] = 0 if state[0] < 0 else state[0]
                    state[1] = 0 if state[1] < 0 else state[1]
                    state[2] = (width - state[0] if state[0] + state[2] > width else state[2])
                    state[3] = (height - state[1] if state[1] + state[3] > height else state[3])

                    cv.rectangle(
                        frame_disp,
                        (state[0], state[1]),
                        (state[2] + state[0], state[3] + state[1]),
                        (0, 0, 255),
                        2,
                    )  # (0, 0, 255)是红色

                    # gt
                    gt = [int(s) for s in anno[f, :]]
                    cv.rectangle(
                        frame_disp,
                        (gt[0], gt[1]),
                        (gt[2] + gt[0], gt[3] + gt[1]),
                        (0, 255, 0),
                        2,
                    )  # 0,255,0是绿色

                    # candidate boxes
                    for i, box in enumerate(frame_candidate_data["candidate_boxes"]):
                        # if i == 0:
                        #     continue
                        temp_state = [int(s) for s in box.squeeze(0)]
                        temp_state[0] = 0 if temp_state[0] < 0 else temp_state[0]
                        temp_state[1] = 0 if temp_state[1] < 0 else temp_state[1]
                        temp_state[2] = (width - temp_state[0] if temp_state[0] +
                                         temp_state[2] > width else temp_state[2])
                        temp_state[3] = (height - temp_state[1] if temp_state[1] +
                                         temp_state[3] > height else temp_state[3])
                        temp_iou = iou(track_result, box.cpu().numpy(), bound=bound)
                        if temp_iou > 0.6:
                            continue
                        # 画出候选框 虚线灰色
                        cv.rectangle(
                            frame_disp,
                            (temp_state[0], temp_state[1]),
                            (
                                temp_state[2] + temp_state[0],
                                temp_state[3] + temp_state[1],
                            ),
                            (200, 200, 200),
                            2,
                        )  # 黄色(0, 255, 255)

                    # 搜索区域
                    search_area_box = frame_candidate_data["search_area_box"]
                    cv.rectangle(
                        frame_disp,
                        (int(search_area_box[0]), int(search_area_box[1])),
                        (
                            int(search_area_box[2]) + int(search_area_box[0]),
                            int(search_area_box[3]) + int(search_area_box[1]),
                        ),
                        (255, 0, 0),
                        2,
                    )
                    save_path = "{}/img/{:>06d}.jpg".format(img_path, f)
                    if not os.path.exists(save_path):
                        makedir(save_path)
                    cv.imwrite(save_path, frame_disp)

                if heatmap_vis:
                    frame_search_img = (frame_candidate_data["search_img"].squeeze().cpu().numpy())

                    # 归一化热力图并转换为颜色映射
                    heatmap = (self.tracker.distractor_dataset_data["score_map"].squeeze().cpu().numpy())
                    heatmap = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX)
                    heatmap = np.uint8(heatmap)
                    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
                    if (heatmap.shape[:2] != frame_search_img.shape[:2]):  # 如果尺寸不一致，调整尺寸
                        heatmap = cv.resize(
                            heatmap,
                            (frame_search_img.shape[1], frame_search_img.shape[0]),
                        )
                    # 融合热力图与原图
                    heat_map_search_img = cv.addWeighted(heatmap, 0.7, frame_search_img, 0.3, 0)

                    # 在图像上添加文本
                    # 通过图像高度动态设置字体大小
                    fontScale = (frame_search_img.shape[1] / 500 * 1.0)  # 基于图像高度动态调整
                    target_candidate_scores = frame_candidate_data["target_candidate_scores"]
                    font_face = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(
                        heat_map_search_img,
                        "No.%06d" % (f),
                        (10, 20),
                        font_face,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    # 将 target_candidate_scores 转换为可显示的字符串
                    if target_candidate_scores.numel() == 1:  # 只有一个元素
                        score_text = "tc_scores: %.2f" % target_candidate_scores.item()
                    else:  # 多个元素，转换为逗号分隔的字符串
                        score_list = (target_candidate_scores.flatten().tolist())  # 转换为 Python 列表
                        score_text = "tc_scores: " + ", ".join(["%.2f" % s for s in score_list])
                    # 在图像上显示分数文本
                    cv.putText(
                        heat_map_search_img,
                        score_text,
                        (10, 50),
                        font_face,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    # 保存图像
                    ca_save_path = "{}/search_img/{:>06d}.jpg".format(img_path, f)
                    if not os.path.exists(ca_save_path):
                        os.makedirs(os.path.dirname(ca_save_path), exist_ok=True)
                    cv.imwrite(ca_save_path, heat_map_search_img)

                if masked:
                    if save_masked_img:
                        masked_save_path = "{}/masked_img/{:>08d}.jpg".format(img_path, f)
                        os.makedirs(os.path.dirname(masked_save_path), exist_ok=True)
                        # save image
                        mask_image_with_boxes(
                            image=image,
                            gt_box=anno[f, :],
                            candidate_boxes=frame_candidate_data["candidate_boxes"],
                            track_result=track_result,
                            iou_threshold=0.7,
                            fill_color=(0, 0, 0),
                            need_save=True,
                            save_path=masked_save_path,
                        )
                    else:
                        masked_save_path = "{}/masked_info.json".format(img_path)
                        process_candidate_boxes_and_gt(
                            seq_name,
                            masked_save_path,
                            frame_id=f,
                            gt=anno[f, :],
                            candidate_boxes=frame_candidate_data["candidate_boxes"],
                            iou_threshold=0.5,
                            bound=bound,
                        )

            # 可选：启用按键中断（例如按'q'退出）
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break

        return seq_candidate_data, scores, boxes
