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
from ..utils.screening_util import *
from ..utils.help import makedir


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
        if self.is_using_cuda:
            print('Detect the CUDA devide')
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
            duration = self._timer_start.elapsed_time(
                self._timer_stop
            )
            duration /= 1000.0
        else:
            duration = time.time() - self._timestamp
        return duration

    def track(self, seq_name, img_files, anno, visualize=False, seq_result_dir=None, save_img=False):
        frame_num = len(img_files)
        box = anno[0, :]  # the information of the first frame
        boxes = np.zeros((frame_num, 4))  # save the tracking result
        boxes[0] = box
        times = np.zeros(frame_num)  # save time

        init_positions = []  # save the restart locations
        if visualize:
            display_name = 'Display: ' + seq_name
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
                    state[2] = width - state[0] if state[0] + state[2] > width else state[2]
                    state[3] = height - state[1] if state[1] + state[3] > height else state[3]
                    font_face = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(frame_disp, 'No.%06d' % (f), (50, 100), font_face, 0.8, (0, 255, 0), 2)
                    if (anno[f, :] != np.array([0, 0, 0, 0])).all():
                        cv.putText(frame_disp, 'seq iou: %2f' % (seq_iou), (50, 130), font_face, 0.8, (0, 255, 0), 2)

                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                 (0, 255, 0), 5)
                    gt = [int(s) for s in anno[f, :]]
                    cv.rectangle(frame_disp, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 0, 255), 5)

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

    def track_for_screening(self, img_files, anno, th=0.1):
        frame_num = len(img_files)
        box = anno[0, :]  # the information of the first frame
        scores = np.zeros((frame_num, 1), dtype=int)  # save the tracking result
        scores[0] = '1'
        seq_candidate_data = defaultdict(list)  # 字典

        for f, img_file in enumerate(img_files):

            image = cv.imread(img_file)
            if f == 0:
                self.init(image, box)
            else:
                _ = self.update(image)  # track

                frame_candidate_data, num = extract_candidate_data(self.tracker.distractor_dataset_data, th=th)
                update_seq_data(seq_candidate_data, frame_candidate_data)

                scores[f] = num

                # print(seq_name, self.name, ' Tracking %d/%d' % (f, frame_num - 1), 'time:%.2f' % times[f], frame_box)

                # key = cv.waitKey(1)
                # if key == ord('q'):
                #     break

        return seq_candidate_data, scores

    def track_for_screening_visual(self, img_files, anno, seq_name, img_path, Is_save, visualize=False):
        frame_num = len(img_files)
        box = anno[0, :]  # the information of the first frame
        boxes = np.zeros((frame_num, 4))  # save the tracking result
        boxes[0] = box

        display_name = 'Display: ' + seq_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)

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
            else:
                frame_box = self.update(image)

                frame_candidate_data, _ = extract_candidate_data(self.tracker.distractor_dataset_data, f)

                frame_box = np.rint(frame_box)

                current_gt = anno[f, :].reshape((1, 4))
                frame_box = np.array(frame_box)
                track_result = frame_box.reshape((1, 4))
                bound = img_resolution
                seq_iou = iou(current_gt, track_result, bound=bound)

                boxes[f, :] = frame_box
                print(seq_name, self.name, ' Tracking %d/%d' % (f, frame_num - 1), frame_box)

                if Is_save:
                    frame_disp = image.copy()
                    state = [int(s) for s in frame_box]
                    state[0] = 0 if state[0] < 0 else state[0]
                    state[1] = 0 if state[1] < 0 else state[1]
                    state[2] = width - state[0] if state[0] + state[2] > width else state[2]
                    state[3] = height - state[1] if state[1] + state[3] > height else state[3]
                    font_face = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(frame_disp, 'No.%06d' % (f), (50, 100), font_face, 0.8, (0, 255, 0), 2)
                    if (anno[f, :] != np.array([0, 0, 0, 0])).all():
                        cv.putText(frame_disp, 'seq iou: %2f' % (seq_iou), (50, 130), font_face, 0.8, (0, 255, 0), 2)

                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                 (0, 255, 0), 2)
                    gt = [int(s) for s in anno[f, :]]
                    cv.rectangle(frame_disp, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 0, 255), 2)
                    # 搜索区域
                    search_area_box = frame_candidate_data['search_area_box']
                    cv.rectangle(frame_disp, (int(search_area_box[0]), int(search_area_box[1])),
                                 (int(search_area_box[2]) + int(search_area_box[0]),
                                  int(search_area_box[3]) + int(search_area_box[1])), (255, 0, 0), 2)

                if Is_save:
                    frame_search_img = frame_candidate_data['search_img'].squeeze().permute(1, 2,
                                                                                            0).cpu().numpy()  # ndarray h,w,c
                    frame_search_img = frame_search_img * 255
                    target_candidate_scores = frame_candidate_data['target_candidate_scores']
                    font_face = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(frame_search_img, 'No.%06d' % (f), (10, 20), font_face, 0.8, (0, 255, 0), 2)
                    cv.putText(frame_search_img, 'tc_scores: %2f' % (target_candidate_scores), (10, 50),
                               font_face, 0.8, (0, 255, 0), 2)

                    heatmap = frame_candidate_data['score_map'].squeeze()
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv.resize(heatmap, (frame_search_img.shape[1], frame_search_img.shape[0]))  # 回归到搜索图片大小
                    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
                    heat_map_search_img = heatmap * 0.5 + frame_search_img

                    # frame_search_img.show()
                if visualize:
                    cv.imshow(display_name, frame_disp)
                if os.path.exists(img_path):
                    if f % 1 == 0:  #
                        save_path = "{}/img/{:>06d}.jpg".format(img_path, f)
                        if not os.path.exists(save_path):
                            makedir(save_path)
                        cv.imwrite(save_path, frame_disp)
                        ca_save_path = "{}/search_img/{:>06d}.jpg".format(img_path, f)
                        if not os.path.exists(ca_save_path):
                            makedir(ca_save_path)
                        cv.imwrite(ca_save_path, heat_map_search_img)
                # key = cv.waitKey(1)
                # if key == ord('q'):
                #     break

        if visualize:
            cv.destroyAllWindows()

        return boxes

    def track_for_score_map(self, img_files, anno):
        frame_num = len(img_files)
        box = anno[0, :]  # the information of the first frame
        seq_score_map_data = defaultdict(list)  # 字典

        for f, img_file in enumerate(img_files):

            image = cv.imread(img_file)
            if f == 0:
                self.init(image, box)
            else:
                _ = self.update(image)  # track
                score_map_data = extract_score_map(self.tracker.distractor_dataset_data)  # score map of frame f
                update_seq_data(seq_score_map_data, score_map_data)  # seq score map dict

                # print(seq_name, self.name, ' Tracking %d/%d' % (f, frame_num - 1), 'time:%.2f' % times[f], frame_box)
                # key = cv.waitKey(1)
                # if key == ord('q'):
                #     break

                # if f == 10:
                #     break

        return seq_score_map_data
