from __future__ import absolute_import, division, print_function

import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import glob
import warnings
import json
from PIL import Image

from ..datasets.votSOI import VOT_SOI
from ..utils.help import makedir
from ..utils.screening_util import *
from ..utils.metrics import rect_iou, center_error, normalized_center_error


def call_back():
    return ExperimentVOTSOI


class ExperimentVOTSOI(object):
    r"""Experiment pipeline and evaluation toolkit for VOT dataset.

    Notes:
        - The tracking results of three types of experiments ``supervised``
            ``unsupervised`` and ``realtime`` are compatible with the official
            VOT toolkit <https://github.com/votchallenge/vot-toolkit/>`.
        - TODO: The evaluation function for VOT tracking results is still
            under development.
    
    Args:
        root_dir (string): Root directory of VOT dataset where sequence
            folders exist.
        version (integer, optional): Specify the VOT dataset version. Specify as
            one of 2013~2018. Default is 2017.
        list_file (string, optional): If provided, only run experiments over
            sequences specified by the file.
        read_image (boolean, optional): If True, return the read PIL image in
            each frame. Otherwise only return the image path. Default is True.
        experiments (string or tuple): Specify the type(s) of experiments to run.
            Default is a tuple (``supervised``, ``unsupervised``, ``realtime``).
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, save_dir, version=2016,
                 read_image=True, list_file=None,
                 experiments='unsupervised',
                 report_dir='reports'):
        super(ExperimentVOTSOI, self).__init__()
        if isinstance(experiments, str):
            experiments = (experiments,)
        assert all([e in ['supervised', 'unsupervised', 'realtime']
                    for e in experiments])
        self.dataset = VOT_SOI(
            root_dir, version, anno_type='rect',
            download=False, return_meta=False, list_file=list_file)
        self.experiments = experiments
        if version == 'LT2018':
            version = '-' + version
        self.read_image = read_image
        self.result_dir = os.path.join(save_dir, 'results')
        self.report_dir = os.path.join(save_dir, 'VOT1619SOI', 'most')
        self.skip_initialize = 5
        self.burnin = 10
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.nbins_nce = 51

    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # run all specified experiments
        if 'unsupervised' in self.experiments:
            self.run_unsupervised(tracker, visualize)

    def run_unsupervised(self, tracker, visualize=False):
        print('Running unsupervised experiment...')

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, 'unsupervised', seq_name,
                '%s_001.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # rectangular bounding boxes
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = self.dataset._corner2rect(anno_rects)

            # tracking loop
            boxes, times = tracker.track(seq_name, img_files, anno_rects, visualize=visualize)
            assert len(boxes) == len(anno)

            # re-formatting
            boxes = list(boxes)
            boxes[0] = [1]

            # record results
            self._record(record_file, boxes, times)

    def _record(self, record_file, boxes, times):
        # convert boxes to string
        lines = []
        for box in boxes:
            if len(box) == 1:
                lines.append('%d' % box[0])
            else:
                lines.append(str.join(',', ['%.4f' % t for t in box]))

        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        with open(record_file, 'w') as f:
            f.write(str.join('\n', lines))
        print('  Results recorded at', record_file)

        # convert times to string
        lines = ['%.4f' % t for t in times]
        lines = [t.replace('nan', 'NaN') for t in lines]

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        if os.path.exists(time_file):
            with open(time_file) as f:
                exist_lines = f.read().strip().split('\n')
            lines = [t + ',' + s for t, s in zip(exist_lines, lines)]
        with open(time_file, 'w') as f:
            f.write(str.join('\n', lines))

    def report(self, tracker_names, th):
        assert isinstance(tracker_names, (list, tuple))
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            if not os.path.exists(os.path.join('/mnt/second/wangyipei/SOI/tracker_result', name, 'votSOI')):
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

            # function for loading results
            def read_record(filename):
                with open(filename) as f:
                    record = f.read().strip().split('\n')
                record = [[float(t) for t in line.split(',')]
                          for line in record]
                return record

            for s, (_, anno) in enumerate(self.dataset):
                # anno = self.dataset._corner2rect(anno)
                seq_name = self.dataset.seq_names[s]
                if seq_name == 'bolt2' or seq_name == 'birds2' or seq_name == 'marching':
                    result_dir = os.path.join('/mnt/second/wangyipei/SOI/tracker_result', name, 'votSOI/test',
                                              'VOT2016')
                else:
                    result_dir = os.path.join('/mnt/second/wangyipei/SOI/tracker_result', name, 'votSOI/test',
                                              self.dataset.name)
                record_file = os.path.join(
                    result_dir, name, '%s.txt' % seq_name)
                # if not exit(record_file):
                #     record_file = os.path.join(
                #         result_dir, name, '%s_%s.txt' % (name, seq_name))
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
                    result_dir, name, '%s_time.txt' % seq_name)
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
               title='Success plots on VOT_SOI')
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
               title='Precision plots on VOT-SOI')
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
               title='Normalized precision plots on VOT-SOI')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving normalized precision plots to', norm_prec_file)
        fig.savefig(norm_prec_file, dpi=300)

