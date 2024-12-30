from __future__ import absolute_import
import argparse
import importlib
import os
from soi.trackers import TrackerODTrack
from soi.local import EnvironmentSettings
import warnings
from soi.utils.analyze_soi_ratios import analyze_retio

import multiprocessing
from itertools import product
import time
from datetime import timedelta
import torch.multiprocessing as mp
import torch


def run_single_sequence_wrapper(experiment, seq_idx, tracker, kwargs):
    """
    包装函数，调用 run_single_sequence，确保参数正确传递。
    """
    try:
        # 获取当前进程名称和ID，分配 GPU
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name.split('-')[-1]) - 1
        num_gpu = torch.cuda.device_count()
        gpu_id = worker_id % num_gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(gpu_id)
        print(f"[Process {worker_name}] 分配到 GPU: {gpu_id}")
    except Exception as e:
        print(f"[Process {multiprocessing.current_process().name}] GPU 分配失败: {e}")
        torch.cuda.set_device(0)  # 回退到 GPU 0

    # 调用实验的单序列运行函数
    experiment.run_single_sequence(seq_idx, tracker, **kwargs)


def choose_tracker(name):
    """根据名称选择追踪器类"""
    tracker_mapping = {
        # 'keeptrack': TrackerKeepTrack,
        # 'ostrack': TrackerOSTrack,
        "odtrack": TrackerODTrack,
    }
    return tracker_mapping.get(name)


def get_list_file(dataset, save_dir, subsets, report_th=None):
    """
    根据不同的数据集配置返回对应的list_file路径
    :param dataset: 数据集名称 ('got10kSOI' 或 'videocubeSOI')
    :param save_dir: 存储文件夹路径
    :param subsets: 子集文件夹名称
    :param report_th: 报告阈值，适用于 'videocubeSOI'
    :return: 对应的数据集list文件路径
    """
    if dataset == "got10kSOI":
        # got10kSOI的数据集配置
        return os.path.join(save_dir, "../data/got10k", subsets, "all-list.txt")
    else:
        return None


def setup_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description="运行目标跟踪器，支持多种数据集和设置。")
    # base settings
    parser.add_argument(
        "--tracker",
        type=str,
        default="odtrack",
        # choices=['tomp', 'srdimp', 'keeptrack',
        # 'ostrack', 'tompKT', 'TransKT', 'uav_kt'],
        help="选择追踪器名称",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lasot",
        choices=[
            "got10k",
            "lasot",
            "otb",
            "vot",
            "votSOI",
            "lasotSOI",
            "got10kSOI",
            "videocube",
            "videocubeSOI",
        ],
        help="选择数据集名称",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/second/wangyipei/SOI/nips25/org_results",
        help="结果保存路径",
    )
    parser.add_argument(
        "--dataset_mkdir",
        type=str,
        default="/mnt/second/wangyipei/SOI/nips25",
        help="数据集根目录",
    )
    parser.add_argument("--subsets", type=str, default="test", help="子数据集名称")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA设备编号")
    # run
    parser.add_argument("--use_filter", action="store_true", help="soi统计")
    parser.add_argument("--threshold", action="store_true", help="soi阈值参数")
    parser.add_argument("--num_threads", default=2, type=int, help="线程数")
    parser.add_argument("--run_mask", action="store_true", help="线程数")
    # report
    parser.add_argument("--report", action="store_true", help="生成报告")
    parser.add_argument("--masked_re", action="store_true", help="测试masked后的序列跟踪性能")
    parser.add_argument("--visual", action="store_true", help="可视化结果")

    return parser


def run_experiment_sequence(sequence, tracker, dataset_dir, save_dir, subsets, list_file):
    """
    在单独的进程中运行单个跟踪序列
    Args:
        sequence: 单个序列或数据集的子集
        tracker: 跟踪器实例
        dataset_dir: 数据集目录
        save_dir: 结果保存目录
        subsets: 数据子集
        list_file: 数据列表文件路径
    """
    try:
        print(f"运行序列: {sequence} 使用跟踪器: {tracker.name}")
        experiment = exper_class(dataset_dir, save_dir, subsets, list_file=list_file)
        experiment.run(tracker)
    except Exception as e:
        print(f"序列 {sequence} 运行失败: {e}")


def main():
    """主函数"""
    warnings.filterwarnings("ignore")
    parser = setup_parser()
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    # 设置CUDA设备
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # print(f"使用的CUDA设备编号: {args.cuda}")

    # 定义保存路径
    save_dir = os.path.join(args.save_dir, args.dataset, args.subsets, args.tracker)
    # tracker_cls = choose_tracker(args.tracker)
    # if tracker_cls is None:
    #     print(f"无效的追踪器名称: {args.tracker}")
    #     return
    # tracker = tracker_cls()

    # ========= 0. pre ========== #
    # 初始化环境设置和实验类
    evs = EnvironmentSettings()
    dataset_dir = evs.find_root_dir(dataset_str=args.dataset)
    # 调用 get_list_file 函数来智能获取 list_file 路径
    list_file = get_list_file(args.dataset, args.save_dir, args.subsets)

    # 加载实验模块
    expr_module = importlib.import_module(f"soi.experiments.{args.dataset}")
    expr_func = getattr(expr_module, "call_back")
    exper_class = expr_func()

    # 根据数据集类型初始化实验
    if args.dataset in ["vot", "votSOI"]:
        experiment = exper_class(dataset_dir, save_dir, version=2019)
    else:
        experiment = exper_class(dataset_dir, save_dir, args.subsets, list_file=list_file)

    # ========= 1. run ========== #
    print("启动多线程追踪器运行...")
    start_time = time.time()

    num_threads = args.num_threads if args.num_threads else multiprocessing.cpu_count()
    # 如果 args.tracker 是单个 tracker，将其转换为列表
    trackers = args.tracker if isinstance(args.tracker, list) else [args.tracker]

    # 遍历所有序列索引和 tracker 的组合
    param_list = [
        (
            experiment,  #
            seq_idx,
            tracker,
            {
                "visualize": args.visual,
                "run_mask": args.run_mask,
                "mask_info": getattr(args, "mask_info", None),
                "use_filter": args.use_filter,
                "threshold": args.threshold,
            },
        ) for seq_idx in range(len(experiment.dataset)) for tracker in trackers
    ]

    print(f"共生成了 {len(param_list)} 个任务，使用 {num_threads} 个线程进行并行处理。")

    # 使用多进程池进行任务调度
    with mp.Pool(processes=num_threads) as pool:
        pool.starmap(run_single_sequence_wrapper, param_list)

    total_time = timedelta(seconds=(time.time() - start_time))
    print(f"多线程追踪完成，总耗时: {total_time}")

    # print("启动追踪器运行...")
    # experiment.run(tracker)

    # ========= 2. report ========== #
    if args.report:
        tracker_names = args.tracker
        # tracker_names = ['TransKT', 'KeepTrack']
        if args.masked_re:
            experiment.report_masked(tracker_names)
        elif args.dataset == "videocubeSOI":
            tracker_names = [name + "_restart" for name in tracker_names]
            experiment.report(tracker_names, args.report_th)
            experiment.report_robust(tracker_names)
        else:
            experiment.report(tracker_names)
        print("报告生成完成")

    # export PYTHONPATH=/home/jaychou/DPcode/SOITrack:$PYTHONPATH


if __name__ == "__main__":
    main()
