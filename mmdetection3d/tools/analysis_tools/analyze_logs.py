import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    # sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')

            if 'AP' in metric:
                xs = []
                ys = []
                for epoch in epochs:
                    if metric not in log_dict[epoch]:
                        continue
                    if len(log_dict[epoch][metric]) > 0:
                        ys += log_dict[epoch][metric]
                        xs.append(int(epoch))

                ax = plt.gca()
                # ax.set_xticks(xs)
                plt.xlabel('epoch')

                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-2]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    if log_dict[epoch][metric] == []:
                        metric = 'center_pos'
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs) / num_iters_per_epoch
                ys = np.concatenate(ys)
                plt.xlabel('epoch')
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()

        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.grid()
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        '--json_logs',
        type=str,
        nargs='+',
        default=[
            #    './work_dirs/smoke/20220430_140722.log.json',
        ],
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=[
            # 'mAP',
            # "img_bbox2d/Easy@0.7@Car@R40@AP2D",
            # "img_bbox2d/Moderate@0.7@Car@R40@AP2D",
            #  "img_bbox2d/Hard@0.7@Car@R40@AP2D",
            "img_bbox/Easy@0.7@Car@R40@AP3D",
            "img_bbox/Moderate@0.7@Car@R40@AP3D",
            "img_bbox/Hard@0.7@Car@R40@AP3D",

            #  "img_bbox/Easy@0.7@Car@R11@AP3D",
            #  "img_bbox/Moderate@0.7@Car@R11@AP3D",
            #  "img_bbox/Hard@0.7@Car@R11@AP3D",
            #  "lr",
            #  "grad_norm",
            #  "roi2_num"
            # 'lhw_error',
            # 'offset_error',
            # "wh_error",
            #  "d_error",
            'iou3d',
            'roi_iou2d',
            #  'iou3d_error',
            # 'iou2d_error',
            # 'iou_error',
            # "uv_error",
            "heatmap_pos",
            #  "heatmap_neg",
            # "iou_iou",
            # "a_error",
            # "sincos_error",
            # "a4bin_error",
            # 'loss'
        ],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


# idea 帧间插值
def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        '--json_logs',
        type=str,
        nargs='+',
        default=['./work_dirs/my_centernet_resnet18_140e_voc0712/20211007_164739.log.json'],
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
             'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log or 'mmdet_version' in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
