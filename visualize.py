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
from utils import get_nodata_value, pick_valid_points

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

    parser.add_argument('--search_dir', action='store_true', help='To search for directories automatically.')

    args = parser.parse_args()
    return args


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def extract_valid_info(log_file, style):
    if 'unloc' in style:
        # pattern = r"Iteration:\s+(?P<iter>\d+), Epoch:\s+(?P<epoch>\d+), Total loss:\s+(?P<ttl_loss>\d+.\d+), " \
        #           r"Task loss: (?P<task_loss>\d+.\d+), SuperCon loss: (?P<supercon_loss>\d+.\d+), Valid: (?P<valid>\d+.\d+)\%, " \
        #           r"Avg Time: (?P<time>\d+.\d+)s"
        if 'supercon' in log_file.lower():
            pattern = r"Iteration:\s+(?P<iter>\d+), Epoch:\s+(?P<epoch>\d+), Total loss:\s+(?P<ttl_loss>\d+.\d+), " \
                      r"Supercon loss:\s+(?P<supercon_loss>\d+.\d+), Valid: (?P<valid>\d+.\d+)\%, " \
                      r"Avg Time: (?P<time>\d+.\d+)s"
        else:
            pattern = r"Iteration:\s+(?P<iter>\d+), Epoch:\s+(?P<epoch>\d+), Total loss:\s+(?P<ttl_loss>\d+.\d+), " \
                      r"Valid: (?P<valid>\d+.\d+)\%, " \
                      r"Avg Time: (?P<time>\d+.\d+)s"
        valid = [[], []]
        with open(log_file, 'r') as file:
            log = file.read()
        for r in re.findall(pattern, log):
            if 'supercon' in log_file.lower():
                valid[0].append(float(r[0]))
                valid[1].append(float(r[4]))
            else:
                valid[0].append(float(r[0]))
                valid[1].append(float(r[3]))
    elif 'dsac' in style:
        if 'log_init' in log_file:
            if 'supercon' in log_file:
                pattern = r"(?P<idx>\d+) (?P<ttlloss>\d+.\d+) (?P<taskloss>\d+.\d+) (?P<superconloss>\d+.\d+) (?P<valid>\d+.\d+)"
            else:
                pattern = r"(?P<idx>\d+) (?P<scrloss>\d+.\d+) (?P<valid>\d+.\d+)"
            valid = []
            with open(log_file, 'r') as file:
                log = file.read()
            if 'supercon' in log_file:
                for r in re.findall(pattern, log):
                    valid.append(float(r[4])*100)
            else:
                for r in re.findall(pattern, log):
                    valid.append(float(r[2])*100)
        elif 'log_e2e' in log_file:
            len_ls = 0
            with open(log_file, 'r') as file:
                for line in file:
                    len_ls += 1
            valid = [0.0] * len_ls
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return valid


def plot_valid_info(valid_ls, log_ls, selected_labels):
    fig_path = 'valid_info.png'
    fig = plt.figure(figsize=(10, 6))
    len_ls = []
    for valid, log in zip(valid_ls, log_ls):
        moving_period = 2000 // 16
        valid_iter = valid[0]
        valid_rate = valid[1]
        valid = moving_average(valid_rate, moving_period)
        valid_iter = np.arange(len(valid)) * np.max(valid_iter) / len(valid)  # re-scaled for moving average
        if 'output.log' in os.path.basename(log):
            label = os.path.basename(os.path.dirname(log))
        else:
            label = os.path.basename(log)
        label = label.replace('comballaz_lhs_sim', '')
        label = label[1:] if label[0] == '_' else label
        finish_loop = False
        for selected_label in selected_labels:
            if selected_label in label or 'all' in selected_label:
                finish_loop = True
                break
        if not finish_loop:
            continue
        if 'debug' in label:
            plt.plot(valid_iter, valid, label=label, linewidth=3, zorder=10)
        elif 'bpnp' in label:
            if 'pose' in label:
                plt.plot(valid_iter, valid, label=label[:-19], linestyle='--')
            else:
                plt.plot(valid_iter, valid, label=label[:-19])
        else:
            plt.plot(valid_iter, valid, label=label)
        len_ls.append(np.max(valid_iter))
    len_ls = np.array(len_ls)
    # plt.xlim([0, np.max(len_ls)])
    # plt.xlim([0, np.mean(len_ls)])
    # plt.xlim([0, 2e6])
    plt.ylabel('Valid SC ratio (%)')
    plt.xlabel('Gradient steps')
    plt.legend(bbox_to_anchor=(1.0, 0.8), loc='center left')
    fig.subplots_adjust(right=0.7)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.show()
    plt.close(fig)
    print('Valid info curve saved to {:s}'.format(fig_path))


def extract_loss_info(log_file, style):
    if 'unloc' in style:
        pattern = r"Iteration:\s+(?P<iter>\d+), Epoch:\s+(?P<epoch>\d+), Total loss:\s+(?P<ttl_loss>\d+.\d+), " \
                  r"Task loss: (?P<task_loss>\d+.\d+), SuperCon loss: (?P<supercon_loss>\d+.\d+), Valid: (?P<valid>\d+.\d+)\%, " \
                  r"Avg Time: (?P<time>\d+.\d+)s"
        loss = []
        scr_loss = []
        supercon_loss = []
        with open(log_file, 'r') as file:
            log = file.read()
        for r in re.findall(pattern, log):
            loss.append(float(r[2]))
            scr_loss.append(float(r[3]))
            supercon_loss.append(float(r[4]))
        return loss, scr_loss, supercon_loss
    elif 'dsac' in style:
        if 'log_init' in log_file:
            pattern = r"(?P<idx>\d+) (?P<scrloss>\d+.\d+) (?P<valid>\d+.\d+)"
        elif 'log_e2e' in log_file:
            pattern = r"(?P<idx>\d+) (?P<scrloss>\d+.\d+)"
        else:
            raise NotImplementedError
        loss = []
        with open(log_file, 'r') as file:
            log = file.read()
        for r in re.findall(pattern, log):
            loss.append(float(r[1]))
        return loss
    else:
        raise NotImplementedError


def plot_loss_info(loss_ls, log_ls, selected_labels):
    fig_path = 'loss_info.png'
    fig = plt.figure(figsize=(10, 6))
    len_ls = []
    for loss, log in zip(loss_ls, log_ls):
        moving_period = 2000 // 16
        loss = moving_average(loss, moving_period)
        if 'output.log' in os.path.basename(log):
            label = os.path.basename(os.path.dirname(log))
        else:
            label = os.path.basename(log)
        label = label.replace('comballaz_lhs_sim', '')
        label = label[1:] if label[0] == '_' else label
        finish_loop = False
        for selected_label in selected_labels:
            if selected_label in label or 'all' in selected_label:
                finish_loop = True
                break
        if not finish_loop:
            continue
        if 'debug' in label:
            plt.plot(np.arange(len(loss)) * moving_period, loss, label=label, linewidth=3, zorder=10)
        elif 'bpnp' in label:
            if 'pose' in label:
                plt.plot(np.arange(len(loss)) * moving_period, loss, label=label[:-19], linestyle='--')
            else:
                plt.plot(np.arange(len(loss)) * moving_period, loss, label=label[:-19])
        else:
            plt.plot(np.arange(len(loss)) * moving_period, loss, label=label)
        len_ls.append(len(loss))
    len_ls = np.array(len_ls)
    plt.xlim([0, np.max(len_ls)])
    loss_median_ls = [np.median(loss) for loss in loss_ls]
    # plt.ylim([0, np.max(loss_median_ls)])
    # plt.ylim([0, 50])
    plt.ylabel('Loss')
    plt.xlabel('Gradient steps')
    plt.legend(bbox_to_anchor=(1.0, 0.8), loc='center left')
    fig.subplots_adjust(right=0.7)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)
    print('Loss curve saved to {:s}'.format(fig_path))


def extract_results_info(log_file, style):
    if 'results' in style:
        pattern = r"(?P<file>[^\s]+) (?P<q_w>[-+]?\d+.\d+) (?P<q_x>[-+]?\d+.\d+) (?P<q_y>[-+]?\d+.\d+) " \
                  r"(?P<q_z>[-+]?\d+.\d+) (?P<t_x>[-+]?\d+.\d+) (?P<t_y>[-+]?\d+.\d+) (?P<t_z>[-+]?\d+.\d+) " \
                  r"(?P<r_err>[-+]?\d+.\d+) (?P<t_err>[-+]?\d+.\d+)"
        r_err_ls, t_err_ls = [], []
        with open(log_file, 'r') as file:
            log = file.read()
        for r in re.findall(pattern, log):
            r_err_ls.append(float(r[8]))  # unit in degree
            t_err_ls.append(float(r[9]))  # unit in meter
        results_ls = [r_err_ls, t_err_ls]
    else:
        return None
    return results_ls


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


def plot_uncertainty_info(unc_map_ls, scene, SUBSAMPLE=8, plot_regress_error=True):
    for unc_map in tqdm(unc_map_ls):
        print('----- Processing uncertainty map at {:s} -----'.format(unc_map))
        unc_dir = os.path.abspath(os.path.join("./datasets/" + scene + "/test", unc_map))
        testset = CamLocDataset("./datasets/" + scene + "/test", mode=1,
                                sparse=True, augment=False, grayscale=False, uncertainty_map=unc_map)
        testset_loader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=6)
        nodata_value = get_nodata_value(scene)
        corr_ls, p_ls = [], []
        for image, gt_pose, init, uncertainty, focal_length, file in tqdm(testset_loader):
            image_ = image.cpu().numpy()[0].transpose((1, 2, 0))  # [H, W, 3]
            if isinstance(uncertainty, torch.Tensor):
                uncertainty_ = uncertainty.cpu().numpy()[0].transpose((1, 2, 0))  # [H, W, 2]
            else:
                uncertainty_ = uncertainty[0].cpu().numpy()[0].transpose((1, 2, 0))  # [H, W, 2]
                t_err, r_err = uncertainty[1].cpu().numpy()[0]

            image = rescale(image_, 1 / SUBSAMPLE, multichannel=True)
            uncertainty = uncertainty_[:, :, 0]
            regress_error = uncertainty_[:, :, 1]

            if plot_regress_error:
                fig = plt.figure(constrained_layout=False, figsize=(7 * 2.5, 7 * 2))
                gs = fig.add_gridspec(nrows=4, ncols=9, hspace=0.2, wspace=0.)
                ax0 = fig.add_subplot(gs[:-1, :])
                ax1 = fig.add_subplot(gs[-1, :2])
                ax2 = fig.add_subplot(gs[-1, 3:5])
                ax3 = fig.add_subplot(gs[-1, 6:8])
                axes = [ax0, ax1, ax2, ax3]
            else:
                fig = plt.figure(figsize=(7 * 1.5, 7))
                plt.subplots_adjust(hspace=-0.1)
                axes = [fig.gca()]

            # first plot for uncertainty
            axes[0].axis('off')
            axes[0].imshow(image)

            unc = axes[0].imshow(uncertainty, cmap='rainbow', alpha=0.5, norm=matplotlib.colors.LogNorm())
            formatter = LogFormatter(base=10.0, labelOnlyBase=False)
            ticks = [1, 5, 10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000, 2000, 3000, 5000]
            if plot_regress_error:
                cbar = plt.colorbar(unc, ax=axes[0], shrink=0.9, ticks=ticks, format=formatter)
            else:
                cbar = plt.colorbar(unc, ax=axes[0], shrink=0.75, ticks=ticks, format=formatter)
            cbar.ax.set_yticklabels((["{:4.0f}".format(i) for i in ticks]))
            unc.set_clim(1, 2000)

            xrng = axes[0].get_xlim()
            yrng = axes[0].get_ylim()
            scale = .25
            offset = 0.1
            axes[0].imshow(image_, extent=(xrng[1] - scale * (xrng[1] - xrng[0]) - offset, xrng[1] - offset,
                                           yrng[1] - scale * (yrng[1] - yrng[0]) - offset, yrng[1] - offset))
            axes[0].set_xlim(xrng)
            axes[0].set_ylim(yrng)

            # second plot for regression loss & uncertainty
            x, y = uncertainty.ravel(), regress_error.ravel()
            gt_coords_valdata = pick_valid_points(init.view(3, -1), nodata_value, boolean=True).cpu().numpy().ravel().astype('bool')
            x, y = x[gt_coords_valdata], y[gt_coords_valdata]
            camera_xyz = gt_pose[0, 0:3, 3]
            depth = torch.norm(init.squeeze(0) - camera_xyz[:, None, None], p=2, dim=0).view(-1)
            z = depth.cpu().numpy()[gt_coords_valdata]

            corr_xy, p_value_xy = stats.spearmanr(x, y)
            corr_xz, p_value_xz = stats.spearmanr(x, z)
            corr_yz, p_value_yz = stats.spearmanr(y, z)

            corr_ls.append(np.array([corr_xy, corr_xz, corr_yz]))
            p_ls.append(np.array([p_value_xy, p_value_xz, p_value_yz]))
            if plot_regress_error:
                axes[0].set_title('Trans. error: {:.2f} m, Rot. error: {:.2f} deg'.format(t_err, r_err), fontsize=24)

                axes[1].scatter(x, y)
                axes[1].set_xlabel('Uncertainty value / sigma (m)', fontsize=18)
                axes[1].set_ylabel('Regression error (m)', fontsize=18)
                axes[1].set_title('Spearman correlation={:.2f}, \n p={:.3f}'.format(corr_xy, p_value_xy), fontsize=20)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_arr = np.linspace(x.min(), x.max(), 100)
                y_arr = intercept + slope * x_arr
                axes[1].plot(x_arr, y_arr, '--r', linewidth=2.5,
                             label='Linear regression (r$^2$={:.2f})'.format(r_value ** 2))
                axes[1].legend(loc='upper left', fontsize=12)

                axes[2].scatter(x, z)
                axes[2].set_xlabel('Uncertainty value / sigma (m)', fontsize=18)
                axes[2].set_ylabel('Depth (m)', fontsize=18)
                axes[2].set_title('Spearman correlation={:.2f}, \n p={:.3f}'.format(corr_xz, p_value_xz), fontsize=20)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, z)
                x_arr = np.linspace(x.min(), x.max(), 100)
                y_arr = intercept + slope * x_arr
                axes[2].plot(x_arr, y_arr, '--r', linewidth=2.5,
                             label='Linear regression (r$^2$={:.2f})'.format(r_value ** 2))
                axes[2].legend(loc='upper left', fontsize=12)

                axes[3].scatter(y, z)
                axes[3].set_xlabel('Regression error (m)', fontsize=18)
                axes[3].set_ylabel('Depth (m)', fontsize=18)
                axes[3].set_title('Spearman correlation={:.2f}, \n p={:.3f}'.format(corr_yz, p_value_yz), fontsize=20)
                slope, intercept, r_value, p_value, std_err = stats.linregress(y, z)
                x_arr = np.linspace(y.min(), y.max(), 100)
                y_arr = intercept + slope * x_arr
                axes[3].plot(x_arr, y_arr, '--r', linewidth=2.5,
                             label='Linear regression (r$^2$={:.2f})'.format(r_value ** 2))
                axes[3].legend(loc='upper left', fontsize=12)

            # save figure
            unc_plot = os.path.join(unc_dir, os.path.basename(file[0]))
            plt.savefig(unc_plot, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)
            # pdb.set_trace()

        corr_ls = np.asarray(corr_ls).transpose()
        p_ls = np.asarray(p_ls).transpose()
        corr_npy = os.path.join(unc_dir, 'correlation_array.npy')
        np.save(corr_npy, np.stack([corr_ls, p_ls]))

        corr_ls, p_ls = np.load(os.path.join(unc_dir, 'correlation_array.npy'))

        p_threshold = 0.01  # very strong evidence
        fig = plt.figure(constrained_layout=False, figsize=(21, 5))
        gs = fig.add_gridspec(nrows=3, ncols=3)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[:, 1])
        ax2 = fig.add_subplot(gs[:, 2])
        axes = [ax0, ax1, ax2]

        for i, (corr, p_value) in enumerate(zip(corr_ls, p_ls)):
            corr = corr[p_value <= p_threshold]
            if i == 0:
                keyword = 'Uncertainty & regression error'
            elif i == 1:
                keyword = 'Uncertainty & depth'
            elif i == 2:
                keyword = 'Regression error & depth'
            print('----- ' + keyword + ' -----')
            print('{:d} out of {:d} instances have very strong evidence against null hypothesis.'.format(len(corr), len(p_value)))
            print('P values data: mean-{:.3f}, median-{:.3f}, min-{:.3f}, max-{:.3f}'.format(
                np.mean(p_value), np.median(p_value), np.min(p_value), np.max(p_value)))

            axes[i].hist(corr, bins=1024, cumulative=True, density=True)
            axes[i].set_ylabel('Cumulative probability of \n {:s}'.format(keyword.lower()), fontsize=14)
            axes[i].set_xlabel('Spearman correlation', fontsize=14)
            corr_median = np.median(corr)
            axes[i].plot([corr_median] * 50, np.linspace(0, 1, 50), label='Median={:.2f}'.format(corr_median))
            corr_mean = np.mean(corr)
            axes[i].plot([corr_mean] * 50, np.linspace(0, 1, 50), label='Mean={:.2f}'.format(corr_mean))
            axes[i].set_ylim([0, 1])
            axes[i].set_xlim([-1, 1])
            axes[i].legend(loc='upper left', fontsize=14)
        plt.savefig(os.path.join(unc_dir, 'correlation_hist.png'), bbox_inches='tight', pad_inches=0.05)

        # split the data in uncertainty folder
        if 'comballaz_lhs_sim' in scene:
            air2_im_0 = sorted([os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if f.endswith('.png') and 'air2' in f and not f.endswith('f2.png')])
            air2_im_1 = sorted([os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if f.endswith('f2.png') and 'air2' in f])
            phantom_p_im = sorted([os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if f.endswith('.png') and 'phantom-piloted' in f])
            phantom_s_im = sorted([os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if f.endswith('.png') and 'phantom-survey' in f])

            def put_in_folder(src_ls, folder_name):
                folder_path = os.path.abspath(os.path.join(unc_dir, folder_name))
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                os.mkdir(folder_path)
                src_ls_ = src_ls
                src_ls = [(src_im, int(os.path.basename(src_im).split('_')[1])) for src_im in src_ls_]

                def take_second(elem):
                    return elem[1]

                src_ls = sorted(src_ls, key=take_second)
                src_ls = [elem[0] for elem in src_ls]
                for i, src_im in enumerate(src_ls):
                    basename = 'img_{:04d}.png'.format(i)
                    target_im = os.path.join(folder_path, basename)
                    os.symlink(src_im, target_im)

            for im_ls, name in zip([air2_im_0, air2_im_1, phantom_p_im, phantom_s_im], ['air2_im_0', 'air2_im_1', 'phantom_p_im', 'phantom_s_im']):
                put_in_folder(im_ls, name)


def plot_depth_info(unc_map_ls, scene, section, SUBSAMPLE=8, plot_regress_error=True):
    for unc_map in tqdm(unc_map_ls):
        print('----- Processing uncertainty map at {:s} -----'.format(unc_map))
        unc_dir = os.path.abspath(os.path.join("./datasets/" + scene + "/{:s}".format(section), unc_map, 'output'))
        os.makedirs(unc_dir, exist_ok=True)
        testset = CamLocDataset("./datasets/" + scene + "/{:s}".format(section), mode=1,
                                sparse=True, augment=False, grayscale=False, raw_image=True)
        testset_loader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=6)
        nodata_value = get_nodata_value(scene)
        corr_ls, p_ls = [], []
        for image, gt_pose, init, focal_length, file in tqdm(testset_loader):
            image_ = image.cpu().numpy()[0].transpose((1, 2, 0))  # [H, W, 3]

            _basename = os.path.basename(os.path.abspath(file[0])).replace('.png', '')
            # pdb.set_trace()
            this_unc_map = os.path.abspath(os.path.join(file[0], '../../', unc_map, _basename + '.npz'))

            assert os.path.exists(this_unc_map)
            unc_map_npz = np.load(this_unc_map)

            pred_map = unc_map_npz['a']
            median_err, mean_err = unc_map_npz['b']

            uncertainty_ = pred_map.transpose((1, 2, 0))  # [H, W, 2]

            image = rescale(image_, 1 / SUBSAMPLE, multichannel=True)
            uncertainty = uncertainty_[:, :, 0]
            depth_error = uncertainty_[:, :, 1]

            if plot_regress_error:
                fig = plt.figure(constrained_layout=False, figsize=(7 * 2.5, 7 * 2))
                gs = fig.add_gridspec(nrows=4, ncols=9, hspace=0.2, wspace=0.)
                ax0 = fig.add_subplot(gs[:-1, :])
                ax1 = fig.add_subplot(gs[-1, :2])
                ax2 = fig.add_subplot(gs[-1, 3:5])
                ax3 = fig.add_subplot(gs[-1, 6:8])
                axes = [ax0, ax1, ax2, ax3]
            else:
                fig = plt.figure(figsize=(7 * 1.5, 7))
                plt.subplots_adjust(hspace=-0.1)
                axes = [fig.gca()]

            # first plot for uncertainty
            axes[0].axis('off')
            axes[0].imshow(image)

            depth_error = depth_error.clip(max=200)
            gt_coords_valdata = pick_valid_points(init.view(1, 3, -1), nodata_value,
                                                  boolean=True).cpu().numpy().ravel().astype('bool')
            gt_coords_nodata = np.logical_not(gt_coords_valdata)
            gt_coords_nodata = gt_coords_nodata.reshape(depth_error.shape)

            depth_error[gt_coords_nodata] = 300.0  # very large error for no-data pixels

            unc = axes[0].imshow(depth_error, cmap='rainbow', alpha=0.5, norm=matplotlib.colors.LogNorm())
            # unc = axes[0].imshow(depth_error, cmap='rainbow', alpha=0.5)
            formatter = LogFormatter(base=10.0, labelOnlyBase=False)
            ticks = [-500, 1, 5, 10, 20, 30, 50, 70, 100, 200, 300, 500]
            if plot_regress_error:
                cbar = plt.colorbar(unc, ax=axes[0], shrink=0.9, ticks=ticks, format=formatter)
            else:
                cbar = plt.colorbar(unc, ax=axes[0], shrink=0.75, ticks=ticks, format=formatter)
            cbar.ax.set_yticklabels((["{:4.0f}".format(i) for i in ticks]))
            unc.set_clim(1, 300)

            xrng = axes[0].get_xlim()
            yrng = axes[0].get_ylim()
            scale = .25
            offset = 0.1
            axes[0].imshow(image_, extent=(xrng[1] - scale * (xrng[1] - xrng[0]) - offset, xrng[1] - offset,
                                           yrng[1] - scale * (yrng[1] - yrng[0]) - offset, yrng[1] - offset))
            axes[0].set_xlim(xrng)
            axes[0].set_ylim(yrng)

            # second plot for regression loss & uncertainty
            x, y = uncertainty.ravel(), depth_error.ravel()

            x, y = x[gt_coords_valdata], y[gt_coords_valdata]

            camera_xyz = gt_pose[0, 0:3, 3]
            gt_depth = torch.norm(init.squeeze(0) - camera_xyz[:, None, None], p=2, dim=0).view(-1)
            z = gt_depth.cpu().numpy()[gt_coords_valdata]

            corr_xy, p_value_xy = stats.spearmanr(x, y)
            corr_xz, p_value_xz = stats.spearmanr(x, z)
            corr_yz, p_value_yz = stats.spearmanr(y, z)

            corr_ls.append(np.array([corr_xy, corr_xz, corr_yz]))
            p_ls.append(np.array([p_value_xy, p_value_xz, p_value_yz]))

            if plot_regress_error:
                axes[0].set_title('Depth estimation median error: {:.2f} m, mean error: {:.2f} m\n the colormap is depth prediction error'.format(
                    median_err, mean_err), fontsize=24)

                axes[1].scatter(x, y)
                axes[1].set_xlabel('Uncertainty value / sigma (m)', fontsize=18)
                axes[1].set_ylabel('Depth regression error (m)', fontsize=18)
                axes[1].set_title('Spearman correlation={:.2f}, \n p={:.4f}'.format(corr_xy, p_value_xy), fontsize=20)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_arr = np.linspace(x.min(), x.max(), 100)
                y_arr = intercept + slope * x_arr
                axes[1].plot(x_arr, y_arr, '--r', linewidth=2.5,
                             label='Linear regression (r$^2$={:.2f})'.format(r_value ** 2))
                axes[1].legend(loc='upper left', fontsize=12)

                axes[2].scatter(x, z)
                axes[2].set_xlabel('Uncertainty value / sigma (m)', fontsize=18)
                axes[2].set_ylabel('Ground-truth depth (m)', fontsize=18)
                axes[2].set_title('Spearman correlation={:.2f}, \n p={:.4f}'.format(corr_xz, p_value_xz), fontsize=20)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, z)
                x_arr = np.linspace(x.min(), x.max(), 100)
                y_arr = intercept + slope * x_arr
                axes[2].plot(x_arr, y_arr, '--r', linewidth=2.5,
                             label='Linear regression (r$^2$={:.2f})'.format(r_value ** 2))
                axes[2].legend(loc='upper left', fontsize=12)

                axes[3].scatter(y, z)
                axes[3].set_xlabel('Depth regression error (m)', fontsize=18)
                axes[3].set_ylabel('Ground-truth (m)', fontsize=18)
                axes[3].set_title('Spearman correlation={:.2f}, \n p={:.4f}'.format(corr_yz, p_value_yz), fontsize=20)
                slope, intercept, r_value, p_value, std_err = stats.linregress(y, z)
                x_arr = np.linspace(y.min(), y.max(), 100)
                y_arr = intercept + slope * x_arr
                axes[3].plot(x_arr, y_arr, '--r', linewidth=2.5,
                             label='Linear regression (r$^2$={:.2f})'.format(r_value ** 2))
                axes[3].legend(loc='upper left', fontsize=12)

            # save figure
            # pdb.set_trace()
            unc_plot = os.path.join(unc_dir, os.path.basename(file[0]))
            plt.savefig(unc_plot, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

        corr_ls = np.asarray(corr_ls).transpose()
        p_ls = np.asarray(p_ls).transpose()
        corr_npy = os.path.join(unc_dir, 'correlation_array.npy')
        np.save(corr_npy, np.stack([corr_ls, p_ls]))

        corr_ls, p_ls = np.load(os.path.join(unc_dir, 'correlation_array.npy'))

        p_threshold = 0.01  # very strong evidence
        fig = plt.figure(constrained_layout=False, figsize=(21, 5))
        gs = fig.add_gridspec(nrows=3, ncols=3)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[:, 1])
        ax2 = fig.add_subplot(gs[:, 2])
        axes = [ax0, ax1, ax2]

        for i, (corr, p_value) in enumerate(zip(corr_ls, p_ls)):
            corr = corr[p_value <= p_threshold]
            if i == 0:
                keyword = 'Uncertainty & regression error'
            elif i == 1:
                keyword = 'Uncertainty & depth'
            elif i == 2:
                keyword = 'Regression error & depth'
            print('----- ' + keyword + ' -----')
            print('{:d} out of {:d} instances have very strong evidence against null hypothesis.'.format(len(corr),
                                                                                                         len(p_value)))
            print('P values data: mean-{:.3f}, median-{:.3f}, min-{:.3f}, max-{:.3f}'.format(
                np.mean(p_value), np.median(p_value), np.min(p_value), np.max(p_value)))

            axes[i].hist(corr, bins=1024, cumulative=True, density=True)
            axes[i].set_ylabel('Cumulative probability of \n {:s}'.format(keyword.lower()), fontsize=14)
            axes[i].set_xlabel('Spearman correlation', fontsize=14)
            corr_median = np.median(corr)
            axes[i].plot([corr_median] * 50, np.linspace(0, 1, 50), label='Median={:.2f}'.format(corr_median))
            corr_mean = np.mean(corr)
            axes[i].plot([corr_mean] * 50, np.linspace(0, 1, 50), label='Mean={:.2f}'.format(corr_mean))
            axes[i].set_ylim([0, 1])
            axes[i].set_xlim([-1, 1])
            axes[i].legend(loc='upper left', fontsize=14)
        plt.savefig(os.path.join(unc_dir, 'correlation_hist.png'), bbox_inches='tight', pad_inches=0.05)

        # split the data in uncertainty folder
        if 'comballaz_lhs_sim' in scene:
            air2_im_0 = sorted([os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if
                                f.endswith('.png') and 'air2' in f and not f.endswith('f2.png')])
            air2_im_1 = sorted(
                [os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if f.endswith('f2.png') and 'air2' in f])
            phantom_p_im = sorted([os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if
                                   f.endswith('.png') and 'phantom-piloted' in f])
            phantom_s_im = sorted(
                [os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if f.endswith('.png') and 'phantom-survey' in f])

            def put_in_folder(src_ls, folder_name):
                folder_path = os.path.abspath(os.path.join(unc_dir, folder_name))
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                os.mkdir(folder_path)
                src_ls_ = src_ls
                src_ls = [(src_im, int(os.path.basename(src_im).split('_')[1])) for src_im in src_ls_]

                def take_second(elem):
                    return elem[1]

                src_ls = sorted(src_ls, key=take_second)
                src_ls = [elem[0] for elem in src_ls]
                for i, src_im in enumerate(src_ls):
                    basename = 'img_{:04d}.png'.format(i)
                    target_im = os.path.join(folder_path, basename)
                    os.symlink(src_im, target_im)

            for im_ls, name in zip([air2_im_0, air2_im_1, phantom_p_im, phantom_s_im],
                                   ['air2_im_0', 'air2_im_1', 'phantom_p_im', 'phantom_s_im']):
                put_in_folder(im_ls, name)

def search_directory(log_path):
    log_path = os.path.abspath(log_path)
    basename = os.path.basename(log_path)
    log_path = os.path.dirname(log_path)

    things_ls = sorted(os.listdir(log_path))
    log_init_ls = [os.path.join(log_path, file) for file in things_ls if file.startswith('log_init') and file.endswith('.txt')]
    log_ls, style_ls = log_init_ls, ['dsac'] * len(log_init_ls)
    for name in things_ls:
        if basename in name and not name.endswith('.txt') and not name.endswith('.net'):
            things_subfolder = sorted(os.listdir(os.path.join(log_path, name)))
            for thing in things_subfolder:
                if 'output.log' in thing:
                    log_ls.append(os.path.join(log_path, name, thing))
                    style_ls.append('unloc')
                    continue
                if ('log_init' in thing or 'log_e2e' in thing) and '.txt' in thing:
                    log_ls.append(os.path.join(log_path, name, thing))
                    style_ls.append('dsac')
                    continue
                if thing.startswith('poses_model_') and thing.endswith('.txt'):
                    log_ls.append(os.path.join(log_path, name, thing))
                    style_ls.append('results')
                    continue
    return log_ls, style_ls


def main(args):
    if args.search_dir:
        log_ls, style_ls = search_directory(args.log_path)
    else:
        log_ls = [args.log_path]
        style_ls = ['unloc']

    # selected_labels = ['all']
    # selected_labels = ['supercon']
    # selected_labels = ['coord']
    # selected_labels = ['depth']
    # selected_labels = ['normal']
    # selected_labels = ['coord-unc-e151-lr0.0002-', '-mlr_depth_normal-resume']
    selected_labels = ['-fullsize']
    # selected_labels = ['-mlr_depth_normal-resume']
    # selected_labels = ['depth-unc-e151-lr0.0002-']
    # selected_labels = ['normal-unc-e151-lr0.0002-']
    # selected_labels = ['debug', 'DEBUG', 'vanilla', 'sim_only']
    # selected_labels = ['DEBUG', 'vanilla', 'sim_only']
    # selected_labels = ['DEBUG_1', 'DEBUG_2', 'DEBUG_3', 'DEBUG_4', 'vanilla', 'sim_only']
    # selected_labels = ['DEBUG_4', 'vanilla', 'sim_only']
    # selected_labels = ['e150', 'e120', 'e151']

    if not isinstance(args.keywords, list):
        args.keywords = [args.keywords]
    log_ls_, style_ls_ = log_ls, style_ls
    for keyword in args.keywords:
        if 'valid' in keyword:
            valid_ls = []
            log_ls = []
            for log, style in zip(log_ls_, style_ls_):
                if 'results' not in style:
                    valid_ls.append(extract_valid_info(log, style))
                    log_ls.append(log)
            plot_valid_info(valid_ls, log_ls, selected_labels)
        elif 'loss' in keyword:
            loss_ls = []
            log_ls = []
            for log, style in zip(log_ls_, style_ls_):
                if 'unloc' in style:
                    loss, _, _ = extract_loss_info(log, style)
                    log_ls.append(log)
                    loss_ls.append(loss)
                elif 'dsac' in style:
                    loss = extract_loss_info(log, style)
                    log_ls.append(log)
                    loss_ls.append(loss)
                else:
                    pass
            plot_loss_info(loss_ls, log_ls, selected_labels)
        elif 'results' in keyword:
            results_ls = []
            for log, style in zip(log_ls, style_ls):
                results_ = extract_results_info(log, style)
                if results_ is not None:
                    results_ls.append(results_)
            plot_results_info(results_ls, log_ls, selected_labels)
        elif 'uncertainty' in keyword:
            scene = os.path.basename(args.log_path)
            unc_map_ls = [f for f in os.listdir("./datasets/" + scene + "/test") if 'calibration' != f and 'init' != f and 'poses' != f and 'rgb' != f]
            SUBSAMPLE = 8
            plot_uncertainty_info(unc_map_ls, scene, SUBSAMPLE, plot_regress_error=True)
        elif 'depth' in keyword:
            scene = os.path.basename(args.log_path)
            section = 'test_drone_sim'
            unc_map_ls = [f for f in os.listdir("./datasets/" + scene + "/{:s}".format(section)) if 'calibration' != f and 'init' != f and 'poses' != f and 'rgb' != f]
            SUBSAMPLE = 8
            plot_depth_info(unc_map_ls, scene, section, SUBSAMPLE, plot_regress_error=True)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    args = parse_argument()
    main(args)

