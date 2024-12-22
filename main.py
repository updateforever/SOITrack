from __future__ import absolute_import
import argparse
import importlib
import os
from soi.trackers import TrackerODTrack
from soi.local import EnvironmentSettings
import warnings
from soi.utils.analyze_soi_ratios import analyze_retio


def choose_tracker(name):
    """根据名称选择追踪器类"""
    tracker_mapping = {
        # 'keeptrack': TrackerKeepTrack,
        # 'ostrack': TrackerOSTrack,
        'odtrack': TrackerODTrack,
    }
    return tracker_mapping.get(name)

def get_list_file(dataset, save_dir, subsets, screen_mode=None, report_th=None):
    """
    根据不同的数据集配置返回对应的list_file路径
    :param dataset: 数据集名称 ('got10kSOI' 或 'videocubeSOI')
    :param save_dir: 存储文件夹路径
    :param subsets: 子集文件夹名称
    :param screen_mode: 屏幕模式，适用于 'videocubeSOI'
    :param report_th: 报告阈值，适用于 'videocubeSOI'
    :return: 对应的数据集list文件路径
    """
    if dataset == 'got10kSOI':
        # got10kSOI的数据集配置
        return os.path.join(save_dir, '../data/got10k', subsets, 'all-list.txt')
    elif dataset == 'videocubeSOI':
        # videocubeSOI的数据集配置
        if screen_mode and report_th:
            return os.path.join(save_dir, '../data/videocube', subsets,
                                f"{screen_mode}th_{report_th}", 'list.txt')
        else:
            raise ValueError("screen_mode and report_th must be provided for 'videocubeSOI' dataset")
    else:
        return None


def setup_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='运行目标跟踪器，支持多种数据集和设置。')
    # base settings
    parser.add_argument('--tracker', type=str, default='odtrack',
                        # choices=['tomp', 'srdimp', 'keeptrack', 'ostrack', 'tompKT', 'TransKT', 'uav_kt'],
                        help='选择追踪器名称')
    parser.add_argument('--dataset', type=str, default='lasot',
                        choices=['got10k', 'lasot', 'otb', 'vot', 'votSOI', 'lasotSOI', 'got10kSOI', 'videocube', 'videocubeSOI'],
                        help='选择数据集名称')
    parser.add_argument('--save_dir', type=str, default='/home/jaychou/DPcode/SOITrack/nips25/org_results',
                        help='结果保存路径')
    parser.add_argument('--dataset_mkdir', type=str, default='/home/jaychou/DPcode/SOITrack/nips25', help='数据集根目录')
    parser.add_argument('--subsets', type=str, default='test', help='子数据集名称')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA设备编号')
    # run mode
    parser.add_argument('--track', action='store_true', help='执行追踪器运行')
    parser.add_argument('--screen_data', action='store_true', help='执行数据筛选任务')
    parser.add_argument('--score_map', action='store_true', help='执行得分映射任务')
    parser.add_argument('--report', action='store_true', help='生成报告')
    parser.add_argument('--masked_re', action='store_true', help='测试masked后的序列跟踪性能')
    # optional settings
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
    save_dir = os.path.join(args.save_dir, args.dataset, args.subsets, args.tracker)
    tracker_cls = choose_tracker(args.tracker)
    if tracker_cls is None:
        print(f"无效的追踪器名称: {args.tracker}")
        return
    tracker = tracker_cls()

    # 初始化环境设置和实验类
    evs = EnvironmentSettings()
    dataset_dir = evs.find_root_dir(dataset_str=args.dataset)
    # 调用 get_list_file 函数来智能获取 list_file 路径
    list_file = get_list_file(args.dataset, args.save_dir, args.subsets, args.screen_mode)

    # 加载实验模块
    expr_module = importlib.import_module(f'soi.experiments.{args.dataset}')
    expr_func = getattr(expr_module, 'call_back')
    exper_class = expr_func()

    # 根据数据集类型初始化实验
    if args.dataset in ['vot', 'votSOI']:
        experiment = exper_class(dataset_dir, save_dir, version=2019)
    else:
        experiment = exper_class(dataset_dir, save_dir, args.subsets, list_file=list_file)

    # 执行选项逻辑
    if args.track:
        print("启动追踪器运行...")
        experiment.run(tracker)
    elif args.screen_data:
        print("启动数据筛选...")
        experiment.run_for_Data_Screening(tracker)
    elif args.score_map:
        print("启动得分图绘制任务...")
        experiment.run_for_score_map(tracker)
    else:
        print("没有指定任务，请选择一个任务来运行：--track, --screen_data, --score_map")

    if args.report:
        tracker_names = args.tracker
        # tracker_names = ['TransKT', 'KeepTrack']
        if args.masked_re:
            experiment.run_masked(tracker)
            experiment.report_masked(tracker_names)
        elif args.dataset == 'videocubeSOI':
            tracker_names = [name + '_restart' for name in tracker_names]
            experiment.report(tracker_names, args.report_th)
            experiment.report_robust(tracker_names)
        else:
            experiment.report(tracker_names)
        print('报告生成完成')

    # if args.make:
    #     experiment.sequences_select(['tomp', 'ostrack', 'srdimp'], args.dataset_mkdir)
    #     print('数据集生成完成')

    # if args.visual:
    #     experiment.run_for_visual(tracker, args.save_visual, args.dataset_mkdir)
    #     print('可视化完成')

    # if args.ratio_analyze:
    #     analyze_retio()
    #     print('SOI比例分析完成')


if __name__ == '__main__':
    main()
