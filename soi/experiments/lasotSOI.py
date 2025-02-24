from __future__ import absolute_import

import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from .base import ExperimentBASE
from ..datasets import LaSOT_SOI
from ..utils.metrics import rect_iou, center_error, normalized_center_error
from ..utils.help import makedir


def call_back():
    return ExperimentLaSOTSOI


class ExperimentLaSOTSOI(ExperimentBASE):
    r"""Experiment pipeline and evaluation toolkit for LaSOT dataset.

    Args:
        root_dir (string): Root directory of LaSOT dataset.
        subset (string, optional): Specify ``train`` or ``test``
            subset of LaSOT.  Default is ``test``.
        return_meta (bool, optional): whether to fetch meta info
        (occlusion or out-of-view).  Default is ``False``.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, save_dir, subset='test', return_meta=False, th=0.1, list_file=None):
        # assert subset.upper() in ['TRAIN', 'TEST']
        self.dataset = LaSOT_SOI(root_dir, subset, return_meta=return_meta,
                                 th=th)  # root_dir = /mnt/second/wangyipei/SOI/lasot/data

        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.nbins_nce = 51

        self.result_dir = os.path.join(save_dir, 'results')
        self.report_dir = os.path.join(save_dir, 'reports')
        self.img_dir = os.path.join(save_dir, 'image')
        makedir(save_dir)
        makedir(self.result_dir)
        makedir(self.report_dir)
        makedir(self.img_dir)

    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            boxes, times = tracker.track(seq_name, img_files, anno, visualize=visualize)
            assert len(boxes) == len(anno)

            # record results
            self._record(record_file, boxes, times)

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def report(self, tracker_names, th=0.1):
        assert isinstance(tracker_names, (list, tuple))
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            if not os.path.exists(os.path.join('/mnt/second/wangyipei/SOI/tracker_result', name, 'lasotSOI')):
                print('no result of %s' % name)
                continue
            print('Evaluating', name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                result_dir = os.path.join(self.result_dir[:self.result_dir.find('SOI') + 3], 'tracker_result', name,
                                          self.dataset.name,
                                          self.result_dir[self.result_dir.find(self.dataset.subset[0]):])

                record_file = os.path.join(
                    result_dir, name, '%s.txt' % seq_name)
                # if not exit(record_file):
                #     record_file = os.path.join(
                #         result_dir, name, '%s_%s.txt' % (name, seq_name))
                if name == str('TransKT') or name == str('ToMP') or name == str('KeepTrack'):  # -MASTER
                    boxes = np.loadtxt(record_file, delimiter='\t')
                else:
                    boxes = np.loadtxt(record_file, delimiter=',')
                boxes[0] = anno[0]
                if not (len(boxes) == len(anno)):
                    # from IPython import embed;embed()
                    print('warning: %s anno donnot match boxes' % seq_name)
                    len_min = min(len(boxes), len(anno))
                    boxes = boxes[:len_min]
                    anno = anno[:len_min]
                assert len(boxes) == len(anno)

                ious, center_errors, norm_center_errors = self._calc_metrics(boxes, anno)
                succ_curve[s], prec_curve[s], norm_prec_curve[s] = self._calc_curves(ious, center_errors,
                                                                                     norm_center_errors)

                # calculate average tracking speed
                time_file = os.path.join(
                    self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'precision_curve': prec_curve[s].tolist(),
                    'normalized_precision_curve': norm_prec_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'precision_score': prec_curve[s][20],
                    'normalized_precision_score': np.mean(norm_prec_curve[s]),
                    'success_rate': succ_curve[s][self.nbins_iou // 2],
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            norm_prec_curve = np.mean(norm_prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            norm_prec_score = np.mean(norm_prec_curve)
            succ_rate = succ_curve[self.nbins_iou // 2]
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'normalized_precision_score': norm_prec_score,
                'success_rate': succ_rate,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        self.plot_curves(tracker_names)

        return performance

    def _calc_metrics(self, boxes, anno):
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            print('Warning: no valid annotations')
            return None, None, None
        else:
            ious = rect_iou(boxes[valid, :], anno[valid, :])
            center_errors = center_error(
                boxes[valid, :], anno[valid, :])
            norm_center_errors = normalized_center_error(
                boxes[valid, :], anno[valid, :])
            return ious, center_errors, norm_center_errors

    def _calc_curves(self, ious, center_errors, norm_center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]
        thr_nce = np.linspace(0, 0.5, self.nbins_nce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)
        bin_nce = np.less_equal(norm_center_errors, thr_nce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)
        norm_prec_curve = np.mean(bin_nce, axis=0)

        return succ_curve, prec_curve, norm_prec_curve

    def plot_curves(self, tracker_names, extension='.png'):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots' + extension)
        prec_file = os.path.join(report_dir, 'precision_plots' + extension)
        norm_prec_file = os.path.join(report_dir, 'norm_precision_plots' + extension)
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on TrackingSOI')  # LaSOT_SOI
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots on TrackingSOI')  # LaSOT-SOI
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)

        # added by user
        # sort trackers by normalized precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['normalized_precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot normalized precision curves
        thr_nce = np.arange(0, self.nbins_nce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_nce,
                            performance[name][key]['normalized_precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['normalized_precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Normalized location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Normalized precision plots on TrackingSOI')  # LaSOT-SOI
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving normalized precision plots to', norm_prec_file)
        fig.savefig(norm_prec_file, dpi=300)

    def plot_curves_soi(self, tracker_names, extension='.png', mode='more'):

        report_dir = os.path.join(self.report_dir, mode)
        report_file_lasot = os.path.join(report_dir, 'performance_lasot.json')
        report_file_vot1619 = os.path.join(report_dir, 'performance_vot1619.json')
        report_file_votlt19 = os.path.join(report_dir, 'performance_votlt19.json')
        report_file_list = [report_file_vot1619, report_file_votlt19, report_file_lasot]

        succ_file = os.path.join(report_dir, mode + '_success_plots' + extension)
        prec_file = os.path.join(report_dir, mode + '_precision_plots' + extension)
        norm_prec_file = os.path.join(report_dir, mode + '_norm_precision_plots' + extension)

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        num_total = {'total': 176, 'once': 56, 'more': 84, 'most': 36}
        num_sot = {'lasot':
                       {'total': 130, 'once': 38, 'more': 62, 'most': 30},
                   'vot1619':
                       {'total': 26, 'once': 6, 'more': 16, 'most': 4},
                   'votlt19':
                       {'total': 20, 'once': 8, 'more': 10, 'most': 2}
                   }
        weight = [26 / 176, 20 / 176, 130 / 176]
        # 定义一个空字典来存储合并后的结果
        performance_result = {}

        for i, report_file in enumerate(report_file_list):
            name = report_file.split('performance_')[1].split('.')[0]
            # load pre-computed performance
            with open(report_file) as f:
                data = json.load(f)
                # 遍历json文件中的每个算法
            for k, v in data.items():
                if k not in performance_result:
                    performance_result[k] = v
                    for key, val in v["overall"].items():
                        if isinstance(val, list):
                            performance_result[k]["overall"][key] = [x * num_sot[name][mode] / num_total[mode]
                                                                     for x in val]
                        else:
                            performance_result[k]["overall"][key] = performance_result[k]["overall"][key] * num_sot[name][mode] / num_total[mode]
                else:
                    # 如果算法在结果字典中已存在，就遍历其overall中的每个键值对，并将其值乘以权重后相加
                    for key, val in v["overall"].items():
                        # 如果值是一个列表，就对列表中的每个元素和结果字典中对应的元素相加
                        if isinstance(val, list):
                            performance_result[k]["overall"][key] = [x + (y * float(num_sot[name][mode]) / num_total[mode]) for x, y in
                                                                     zip(performance_result[k]["overall"][key], val)]
                        else:
                            performance_result[k]["overall"][key] += val * num_sot[name][mode] / num_total[mode]

        performance = performance_result
        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        key = 'overall'
        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on TrackingSOI')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots on TrackingSOI')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)

        # added by user
        # sort trackers by normalized precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['normalized_precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot normalized precision curves
        thr_nce = np.arange(0, self.nbins_nce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_nce,
                            performance[name][key]['normalized_precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['normalized_precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Normalized location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Normalized precision plots on TrackingSOI')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving normalized precision plots to', norm_prec_file)
        fig.savefig(norm_prec_file, dpi=300)
