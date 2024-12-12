from __future__ import absolute_import
import argparse
import importlib
import os
from soi.experiments import ExperimentLaSOTSOI
from soi.trackers import *
from soi.local import EnvironmentSettings
import warnings
from soi.utils.analyze_soi_ratios import analyze_retio


def choose_tracker(name):
    """根据名称选择追踪器类"""
    tracker_mapping = {
        'tomp': TrackerToMP,
        'keeptrack': TrackerKeepTrack,
        'tompKT': TrackerToMPKT,
        'srdimp': TrackerSuperDiMP,
        'TransKT': TrackerTransKT,
        'uav_kt': TrackerUAVKT
    }
    return tracker_mapping.get(name)


def setup_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='运行目标跟踪器，支持多种数据集和设置。')
    parser.add_argument('--tracker', type=str, default='ostrack',
                        choices=['tomp', 'srdimp', 'keeptrack', 'ostrack', 'tompKT', 'TransKT', 'uav_kt'],
                        help='选择追踪器名称')
    parser.add_argument('--dataset', type=str, default='vot',
                        choices=['got10k', 'lasot', 'otb', 'vot', 'votSOI', 'lasotSOI', 'got10kSOI', 'videocube', 'videocubeSOI'],
                        help='选择数据集名称')
    parser.add_argument('--save_dir', type=str, default='/mnt/second/wangyipei/SOI/tcsvt_version/TrackingSOI/org_results',
                        help='结果保存路径')
    parser.add_argument('--dataset_mkdir', type=str, default='/mnt/second/wangyipei/SOI', help='数据集根目录')
    parser.add_argument('--subsets', type=str, default='test', help='子数据集名称')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA设备编号')
    parser.add_argument('--run', action='store_true', help='执行追踪器运行')
    parser.add_argument('--run_s', action='store_true', help='执行SOI筛选运行')
    parser.add_argument('--runs_th', type=float, default=0.3,
                        choices=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], help='筛选得分阈值')
    parser.add_argument('--run_score_map', action='store_true', help='运行得分地图生成')
    parser.add_argument('--report', action='store_true', help='生成报告')
    parser.add_argument('--report_th', type=float, default=0.3,
                        choices=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], help='报告得分阈值')
    parser.add_argument('--make', action='store_true', help='生成数据集')
    parser.add_argument('--screen_mode', type=str, default='MaxPooling',
                        choices=['MaxPooling', 'MAX_SCORE'], help='筛选模式')
    parser.add_argument('--visual', action='store_true', help='可视化结果')
    parser.add_argument('--save_visual', action='store_true', help='保存可视化结果')
    parser.add_argument('--ratio_analyze', action='store_true', help='分析SOI比例')
    return parser


def main():
    """主函数"""
    warnings.filterwarnings("ignore")
    parser = setup_parser()
    args = parser.parse_args()

    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f'使用的CUDA设备编号: {args.cuda}')

    # 定义保存路径
    save_dir = os.path.join(args.save_dir, args.tracker, args.dataset, args.subsets)
    tracker_cls = choose_tracker(args.tracker)
    if tracker_cls is None:
        print(f"无效的追踪器名称: {args.tracker}")
        return
    tracker = tracker_cls()

    # 初始化环境设置和实验类
    evs = EnvironmentSettings()
    dataset_dir = evs.find_root_dir(dataset_str=args.dataset)
    list_file = None

    # 根据数据集配置list_file
    if args.dataset == 'got10kSOI':
        list_file = os.path.join(args.save_dir, '../data/got10k', args.subsets, 'all-list.txt')
    elif args.dataset == 'videocubeSOI':
        list_file = os.path.join(args.save_dir, '../data/videocube', args.subsets,
                                 f"{args.screen_mode}th_{args.report_th}", 'list.txt')

    # 加载实验模块
    expr_module = importlib.import_module(f'soi.experiments.{args.dataset}')
    expr_func = getattr(expr_module, 'call_back')
    exper_class = expr_func()

    # 根据数据集类型初始化实验
    if args.dataset in ['vot', 'votSOI']:
        experiment = exper_class(dataset_dir, save_dir, version=2019)
    else:
        experiment = exper_class(dataset_dir, save_dir, args.subsets, th=args.report_th, list_file=list_file)

    # 执行选项逻辑
    if args.run_s:
        experiment.run_for_Data_Screening(tracker, args.runs_th)
        print('数据筛选运行完成')
    elif args.run:
        experiment.run(tracker)
        print('追踪器运行完成')
    elif args.run_score_map:
        experiment.run_for_score_map(tracker)
        print('得分地图生成完成')

    if args.report:
        tracker_names = ['TransKT', 'KeepTrack']
        if args.dataset == 'videocubeSOI':
            tracker_names = [name + '_restart' for name in tracker_names]
            experiment.report(tracker_names, args.report_th)
            experiment.report_robust(tracker_names)
        else:
            experiment.report(tracker_names)
        print('报告生成完成')

    if args.make:
        experiment.sequences_select(['tomp', 'ostrack', 'srdimp'], args.dataset_mkdir)
        print('数据集生成完成')

    if args.visual:
        experiment.run_for_visual(tracker, args.save_visual, args.dataset_mkdir)
        print('可视化完成')

    if args.ratio_analyze:
        analyze_retio()
        print('SOI比例分析完成')


if __name__ == '__main__':
    main()
