import argparse
import pdb
import re
import os
import shutil
from tqdm import tqdm

from scipy import stats
from skimage.transform import rescale

import torch
from dataloader.dataloader import CamLocDataset
from utils.learning import get_nodata_value, pick_valid_points

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter


def parse_argument():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('log_path', type=str, help='Logging file path.')

    parser.add_argument('--keywords', type=str, default=None, nargs='+', help='Visualize keyword values from log.')

    args = parser.parse_args()
    return args


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def extract_training_info(log_file, task, reproj_loss=False):
    if task in ['coord', 'depth', 'normal', 'semantics']:
        # general training meta data
        pattern = r"Iteration:\s+(?P<iter>\d+), Epoch:\s+(?P<epoch>\d+), Total loss:\s+-?(?P<ttl_loss>\d+.\d+), " \
                  r"Valid: (?P<valid>\d+.\d+)\%, " \
                  r"Avg Time: (?P<time>\d+.\d+)s"
        training_iter, training_epoch, training_loss, valid_rate = [], [], [], []
        with open(log_file, 'r') as file:
            log = file.read()
        for r in re.findall(pattern, log):
            training_iter.append(float(r[0]))
            training_epoch.append(int(r[1]))
            training_loss.append(float(r[2]))
            valid_rate.append(float(r[3]))

        # specific task error
        if task == 'coord':
            pattern = r"Regression error: coord:\s+-?(?P<reg_error>\d+.\d+), reprojection:\s+-?(?P<reproj_error>\d+.\d+)"
            task_error = [[], []]
            for r in re.findall(pattern, log):
                task_error[0].append(float(r[0]))
                task_error[1].append(float(r[1]))
            task_error = task_error[1] if reproj_loss else task_error[0]
        elif task == 'depth':
            pattern = r"Regression error: depth:\s+-?(?P<reg_error>\d+.\d+)"
            task_error = []
            for r in re.findall(pattern, log):
                task_error.append(float(r[0]))
        elif task == 'normal':
            pattern = r"Regression error: normal in radian:\s+-?(?P<reg_error_radian>\d+.\d+), " \
                      r"normal in degree:\s+-?(?P<reg_error_deg>\d+.\d+)"
            task_error = []
            for r in re.findall(pattern, log):
                task_error.append(float(r[1]))
        else:
            task_error = None
    else:
        raise NotImplementedError
    return training_iter, training_epoch, training_loss, task_error, valid_rate


def plot_training_info(training_info_ls, log_ls, mode='valid', show_epoch=False, session_name=''):
    # training_info is a list of [training_iter, training_epoch, training_loss, task_error, valid_rate]
    moving_period = 2000 // 16  # hyper-parameter
    fig_path = 'training_info_{:s}.png'.format(session_name) if session_name is not None else 'training_info.png'
    fig = plt.figure(figsize=(10, 6))
    len_ls, stat_y_min, stat_y_max, stat_y_mean, stat_y_std = [], [], [], [], []
    for training_info, log in zip(training_info_ls, log_ls):
        label = os.path.basename(os.path.dirname(log))

        training_iter, training_epochs, training_loss, task_error, valid_rate = training_info

        training_loss = moving_average(training_loss, moving_period)
        valid_rate = moving_average(valid_rate, moving_period)
        task_error = moving_average(task_error, moving_period) if task_error is not None else task_error

        # re-scaled for moving average
        training_iter = np.arange(len(valid_rate)) * np.max(training_iter) / len(valid_rate)
        training_epochs = np.linspace(0, np.max(training_epochs), len(valid_rate))

        if mode == 'loss':
            y_data = training_loss
        elif mode == 'valid':
            y_data = valid_rate
        elif mode == 'task_error':
            if task_error is None:
                continue
            y_data = task_error
        else:
            raise NotImplementedError

        stat_y_min.append(np.min(y_data))
        stat_y_max.append(np.max(y_data))
        stat_y_mean.append(np.mean(y_data))
        stat_y_std.append(np.std(y_data))

        if show_epoch:
            x_data = training_epochs
        else:
            x_data = training_iter

        if len(y_data) > len(x_data):
            y_data = y_data[0:len(x_data)]
        else:
            x_data = x_data[0:len(y_data)]

        if 'debug' in label:
            plt.plot(x_data, y_data, label=label, linewidth=3, zorder=10)
        else:
            plt.plot(x_data, y_data, label=label)
        len_ls.append(np.max(training_iter))
    len_ls = np.array(len_ls)
    # plt.xlim([0, np.max(len_ls)])
    # if mode != 'valid':
    #     plt.ylim([0, 20])
    # plt.ylim([np.min(stat_y_min), np.mean(stat_y_mean) + 3.0 * np.mean(stat_y_std)])

    if mode == 'valid':
        plt.ylabel('Valid pixel ratio (%)')
        plt.ylim([plt.gca().get_ylim()[0], min(plt.gca().get_ylim()[1], 100)])
    elif mode == 'loss':
        plt.ylabel('Loss value')
    elif mode == 'task_error':
        plt.ylabel("Task specific error")
    else:
        raise NotImplementedError

    if show_epoch:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Gradient steps')
    plt.legend(bbox_to_anchor=(1.0, 0.8), loc='center left')
    fig.subplots_adjust(right=0.7)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    # plt.show()
    plt.close(fig)
    print('Curve saved to {:s}'.format(fig_path))


def plot_results_info(results_ls, log_ls, selected_labels):
    fig_path = 'results_info.png'
    len_ls = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    colors_ls = matplotlib.cm.prism(np.linspace(0, 1, len(log_ls)))
    for i, ((r_err, t_err), log) in enumerate(zip(results_ls, log_ls)):
        label = os.path.basename(log).replace('comballaz_lhs_sim', '').replace('poses_model_', '').replace('.txt', '').replace('__', '')
        label = label.replace('comballaz_lhs_init', '').replace('.pth', '').replace('.net', '')
        label = label[1:] if label[0] == '_' else label
        label = ''
        finish_loop = False
        for selected_label in selected_labels:
            if selected_label in label or 'all' in selected_label:
                finish_loop = True
                break
        if not finish_loop:
            continue
        axes[0].hist(r_err, label=label, bins=1024, cumulative=True, density=True, color=colors_ls[i], alpha=0.3)
        axes[1].hist(t_err, label=label, bins=1024, cumulative=True, density=True, color=colors_ls[i], alpha=0.3)
        axes[0].set_ylim([0, 1])
        axes[1].set_ylim([0, 1])


        # axes[0].hist(r_err, label=label, bins=1024, alpha=0.5, color=colors_ls[i])
        # axes[1].hist(t_err, label=label, bins=1024, alpha=0.5, color=colors_ls[i])

        axes[0].plot([np.median(r_err)]*50, np.linspace(*axes[0].get_ylim(), 50),
                     label=label+'Median={:.1f}deg'.format(np.median(r_err)), color=colors_ls[i])
        axes[1].plot([np.median(t_err)]*50, np.linspace(*axes[1].get_ylim(), 50),
                     label=label+'Median={:.1f}m'.format(np.median(t_err)), color=colors_ls[i])
        len_ls.append(len(r_err))
    axes[0].set_xlim([0, 15])
    axes[1].set_xlim([0, 100])
    axes[0].set_xlabel('Rotational error (deg)', fontsize=18)
    axes[1].set_xlabel('Translational error (m)', fontsize=18)
    axes[0].set_ylabel('Cumulative probability', fontsize=18)
    axes[1].set_ylabel('Cumulative probability', fontsize=18)
    # axes[0].legend(bbox_to_anchor=(1.0, 0.8), loc='center left')
    # axes[1].legend(bbox_to_anchor=(1.0, 0.8), loc='center left')
    axes[0].legend(loc='center right', fontsize=18)
    axes[1].legend(loc='center right', fontsize=18)
    fig.subplots_adjust(wspace=0.3)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close(fig)
    print('Results info curve saved to {:s}'.format(fig_path))


def search_directory(log_path, keywords):
    """Search for the log files."""

    if os.path.isdir(log_path):
        assert keywords is not None, "Keywords must be enabled when log_path is a file!"
        log_path = os.path.abspath(log_path)
        log_path_ls = []
        for root, dirs, files in os.walk(log_path):
            for file in files:
                if 'output.log' in file:
                    flag_keyword = all([keyword in os.path.join(root, file) for keyword in keywords])
                    if flag_keyword:
                        log_path_ls.append(os.path.join(root, file))
    elif os.path.isfile(log_path):
        assert keywords is None, "Keywords must be disabled when log_path is a file!"
        assert 'output.log' in os.path.basename(log_path), "log file {:s} not supported!".format(log_path)
        log_path_ls = [log_path]
    else:
        raise NotImplementedError

    log_path_ls = sorted(log_path_ls)

    tasks_ls = []
    for log in log_path_ls:
        if 'coord' in os.path.dirname(log):
            tasks_ls.append('coord')
        elif 'depth' in os.path.dirname(log):
            tasks_ls.append('depth')
        elif 'normal' in os.path.dirname(log):
            tasks_ls.append('normal')
        elif 'semantics' in os.path.dirname(log):
            tasks_ls.append('semantics')
        else:
            raise NotImplementedError

    print("With keywords {}, {:d} log files are found.".format(keywords, len(log_path_ls)))
    return log_path_ls, tasks_ls


def main():
    args = parse_argument()

    log_ls, task_ls = search_directory(args.log_path, args.keywords)

    # manually add some selection criteria
    selected_labels = ['all']
    # selected_labels = ['coord']
    # selected_labels = ['depth']
    # selected_labels = ['normal']
    # selected_labels = ['finetune']
    # selected_labels = ['real_only']
    # selected_labels = ['oop-rc1.00-finetune']
    # selected_labels = ['pairs-ip-rc1.0']
    # selected_labels = ['decoder']
    # selected_labels = ['sim_only']
    # selected_labels = ['sc0.25']
    # selected_labels = ['sc0.50']
    # selected_labels = ['sc0.75']

    _log_ls, _task_ls = log_ls.copy(), task_ls.copy()
    log_ls, task_ls = [], []
    for log, task in zip(_log_ls, _task_ls):
        flag_select = any([keyword in log for keyword in selected_labels])
        flag_select = True if 'all' in selected_labels else flag_select
        if flag_select:
            log_ls.append(log)
            task_ls.append(task)
    assert len(log_ls), "There is no available logs to read and plot!"

    reproj_loss = False

    # obtain valid rate
    training_info_ls = []
    for log, task in zip(log_ls, task_ls):
        # result is a tuple, training_iter, training_epoch, training_loss, task_error, valid_rate
        training_info_ls.append(extract_training_info(log, task, reproj_loss))

    # make the plot
    plot_training_info(training_info_ls, log_ls, mode='valid', show_epoch=False, session_name='valid_by_iter')
    plot_training_info(training_info_ls, log_ls, mode='valid', show_epoch=True, session_name='valid_by_epoch')
    plot_training_info(training_info_ls, log_ls, mode='loss', show_epoch=False, session_name='loss_by_iter')
    plot_training_info(training_info_ls, log_ls, mode='loss', show_epoch=True, session_name='loss_by_epoch')
    plot_training_info(training_info_ls, log_ls, mode='task_error', show_epoch=False, session_name='task_error_by_iter')
    plot_training_info(training_info_ls, log_ls, mode='task_error', show_epoch=True, session_name='task_error_by_epoch')


if __name__ == '__main__':
    main()

