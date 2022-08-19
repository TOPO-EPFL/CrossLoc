import torch
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset

import shutil
import argparse
import math
import os
import pdb
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

PROJECT_DIR = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, PROJECT_DIR)
from dataloader import FLAG_ANCHOR, FLAG_POS_IN_DOM, FLAG_POS_CROSS_DOM, FLAG_NEG_IN_DOM, FLAG_NEG_CROSS_DOM
from dataloader.dataloader import CamLocDataset
from dataloader.dataloader_supercon import CamLocDatasetSupercon
from utils import get_unique_file_name, load_geo_dist, get_supercon_dataloader
from datasets.setup_geo_distance import compute_raw_ranking


def config_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', help='name of a scene in the dataset folder')

    parser.add_argument('--real_chunk', type=float, default=1.0,
                        help='to chunk the real data with given proportion')

    opt = parser.parse_args()
    return opt


def config_dataloader(scene, top_n, real_chunk=1.0):
    """
    Config dataloader for original dataset w/o augmentation, used for initialization.
    """
    if real_chunk == 1.0:
        root_real = os.path.join(PROJECT_DIR, 'datasets', scene, 'train_real')
    else:
        root_real = os.path.join(PROJECT_DIR, 'datasets', scene, 'train_real_chunk_{:.2f}'.format(real_chunk))
    trainset = CamLocDataset([os.path.join(PROJECT_DIR, 'datasets', scene, 'train_sim_aug'),
                              root_real], mode=1,
                             sparse=True, augment=False, grayscale=False, raw_image=True,
                             real_chunk=real_chunk)

    sampling_pos_cross_dom_top_n = top_n - 1
    sampling_pos_in_dom_top_n = top_n
    sampling_neg_cross_dom_top_n = top_n
    sampling_neg_in_dom_top_n = top_n

    trainset_supercon = CamLocDatasetSupercon(root_dir_sim=os.path.join(PROJECT_DIR, "datasets", scene, "train_sim_aug"),
                                              root_dir_real=root_real,
                                              augment=False, grayscale=False, raw_image=True, real_chunk=real_chunk,
                                              supercon=True,
                                              sampling_pos_cross_dom_top_n=sampling_pos_cross_dom_top_n,
                                              sampling_pos_in_dom_top_n=sampling_pos_in_dom_top_n,
                                              sampling_neg_cross_dom_top_n=sampling_neg_cross_dom_top_n,
                                              sampling_neg_in_dom_top_n=sampling_neg_in_dom_top_n)

    trainset_loader = get_supercon_dataloader(trainset_supercon, shuffle=False)  # no need to shuffle for first epoch

    return trainset, trainset_supercon, trainset_loader


def get_img_from_dataloader(trainset, idx):
    """
    Fast low-level implementation to fetch an image.
    """
    file_path = trainset[idx][-1]
    assert os.path.exists(file_path)
    return io.imread(file_path)[:, :, :3]  # [H, W, 3]


def get_distances(trainset, idx_a, idx_b):
    """
    Compute translational and rotational distances between two camera poses.
    Warning: this function should not be used for batch processing.
    """
    pose_a, pose_b = trainset[idx_a][1].view(4, 4).cpu().numpy(), trainset[idx_b][1].view(4, 4).cpu().numpy()
    r_err = np.matmul(pose_a[0:3, 0:3], np.transpose(pose_b[0:3, 0:3]))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi  # unit: degree

    t_err = float(np.linalg.norm(pose_a[0:3, 3] - pose_b[0:3, 3]))
    return t_err, r_err


def get_rank_by_sample(dict_ranking, same_domain, anchor_id, sample_id):
    """
    Get the geometric distance ranking of the anchor seen from the sample.
    """
    # extract argsort index list
    if same_domain:
        dist_argsort_ = dict_ranking[sample_id]['same']
    else:
        dist_argsort_ = dict_ranking[sample_id]['diff']
    # find ranking
    if anchor_id not in dist_argsort_:
        pdb.set_trace()

    assert anchor_id in dist_argsort_
    return int(np.where(dist_argsort_ == anchor_id)[0])


def _create_folder(scene, real_chunk):
    preview_dir = os.path.join(os.path.join(PROJECT_DIR, "datasets", scene, 'preview_chunk_{:.2f}'.format(real_chunk)))
    if os.path.exists(preview_dir):
        shutil.rmtree(preview_dir)
    os.makedirs(preview_dir)
    os.makedirs(os.path.join(preview_dir, 'feasible_sim_anchor'))
    os.makedirs(os.path.join(preview_dir, 'feasible_real_anchor'))
    os.makedirs(os.path.join(preview_dir, 'infeasible_sim_anchor'))
    os.makedirs(os.path.join(preview_dir, 'infeasible_real_anchor'))
    return preview_dir


def plot_preview(preview_dir, trainset, batch_identities, dict_inv_ranking, sim_data, dict_name_to_idx, feasible_anchor, top_n, images, file_names):
    """
    Plot and save the preview for qualitative results.
    """
    anchor_id = dict_name_to_idx[get_unique_file_name(file_names.reshape(-1)[0])]
    is_sim = sim_data[anchor_id]

    if anchor_id in feasible_anchor:
        # feasible anchor
        mask_anchor = np.equal(batch_identities, FLAG_ANCHOR)
        mask_pos_cross_dom = np.equal(batch_identities, FLAG_POS_CROSS_DOM)
        mask_pos_in_dom = np.equal(batch_identities, FLAG_POS_IN_DOM)
        mask_neg_cross_dom = np.equal(batch_identities, FLAG_NEG_CROSS_DOM)
        mask_neg_in_dom = np.equal(batch_identities, FLAG_NEG_IN_DOM)

        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        anchor_img = images[mask_anchor]
        pos_cross_dom_img = images[mask_pos_cross_dom][:top_n-1]
        pos_in_dom_img = images[mask_pos_in_dom][:top_n]
        neg_cross_dom_img = images[mask_neg_cross_dom][:top_n]
        neg_in_dom_img = images[mask_neg_in_dom][:top_n]

        anchor_file = file_names[mask_anchor]
        pos_cross_dom_files = file_names[mask_pos_cross_dom][:top_n-1]
        pos_in_dom_files = file_names[mask_pos_in_dom][:top_n]
        neg_cross_dom_files = file_names[mask_neg_cross_dom][:top_n]
        neg_in_dom_files = file_names[mask_neg_in_dom][:top_n]

        # show images
        fig, axes = plt.subplots(4, top_n, figsize=(25, 12))
        for i in range(4):
            for j in range(top_n):
                axes[i, j].axis('off')
        axes[0, 0].imshow(anchor_img[0])

        for i, cd_pos in enumerate(pos_cross_dom_img):
            axes[0, 1 + i].imshow(cd_pos)

        for i, id_pos in enumerate(pos_in_dom_img):
            axes[1, i].imshow(id_pos)

        for i, cd_neg in enumerate(neg_cross_dom_img):
            axes[2, i].imshow(cd_neg)

        for i, id_neg in enumerate(neg_in_dom_img):
            axes[3, i].imshow(id_neg)

        axes[0, 0].set_title('Anchor ({:s})'.format('sim' if is_sim else 'real'))
        for i, cd_pos_file in enumerate(pos_cross_dom_files):
            cd_pos_id = dict_name_to_idx[get_unique_file_name(cd_pos_file)]
            t_err, r_err = get_distances(trainset, anchor_id, cd_pos_id)
            if sim_data[cd_pos_id] == sim_data[anchor_id]:
                pdb.set_trace()

            assert sim_data[cd_pos_id] != sim_data[anchor_id], "Anchor: {:d} SIM is {}, CD-POS: {:d} SIM is {}".format(anchor_id, sim_data[anchor_id], cd_pos_id, sim_data[cd_pos_id])
            assert anchor_id in dict_inv_ranking[cd_pos_id]['diff'], 'Anchor: {:d} is not in the "diff" INV rank list of sample {:d}'.format(anchor_id, cd_pos_id)
            anchor_rank = dict_inv_ranking[cd_pos_id]['diff'][anchor_id]
            axes[0, 1 + i].set_title(
                'Cross-dom. POS Top-{:d}, rank-{:d}\n Dist: {:.1f} m, {:.1f} deg'.format(i + 1, anchor_rank + 1, t_err,
                                                                                         r_err))

        for i, id_pos_file in enumerate(pos_in_dom_files):
            id_pos_id = dict_name_to_idx[get_unique_file_name(id_pos_file)]
            t_err, r_err = get_distances(trainset, anchor_id, id_pos_id)
            assert sim_data[id_pos_id] == sim_data[anchor_id], "Anchor: {:d} SIM is {}, ID-POS: {:d} SIM is {}".format(anchor_id, sim_data[anchor_id], id_pos_id, sim_data[id_pos_id])
            assert anchor_id in dict_inv_ranking[id_pos_id]['same'], 'Anchor: {:d} is not in the "same" INV rank list of sample {:d}'.format(anchor_id, id_pos_id)
            anchor_rank = dict_inv_ranking[id_pos_id]['same'][anchor_id]
            axes[1, i].set_title(
                'In-dom. POS Top-{:d}, rank-{:d}\n Dist: {:.1f} m, {:.1f} deg'.format(i + 1, anchor_rank + 1, t_err,
                                                                                      r_err))

        for i, cd_neg_file in enumerate(neg_cross_dom_files):
            cd_neg_id = dict_name_to_idx[get_unique_file_name(cd_neg_file)]
            t_err, r_err = get_distances(trainset, anchor_id, cd_neg_id)
            assert sim_data[cd_neg_id] != sim_data[anchor_id], "Anchor: {:d} SIM is {}, CD-NEG: {:d} SIM is {}".format(anchor_id, sim_data[anchor_id], cd_neg_id, sim_data[cd_neg_id])
            assert anchor_id in dict_inv_ranking[cd_neg_id]['diff'], 'Anchor: {:d} is not in the "diff" INV rank list of sample {:d}'.format(anchor_id, cd_neg_id)
            anchor_rank = dict_inv_ranking[cd_neg_id]['diff'][anchor_id]
            axes[2, i].set_title(
                'Cross-dom. NEG Top-{:d}, rank-{:d}\n Dist: {:.1f} m, {:.1f} deg'.format(i + 1, anchor_rank + 1, t_err,
                                                                                         r_err))

        for i, id_neg_file in enumerate(neg_in_dom_files):
            id_neg_id = dict_name_to_idx[get_unique_file_name(id_neg_file)]
            t_err, r_err = get_distances(trainset, anchor_id, id_neg_id)
            assert sim_data[id_neg_id] == sim_data[anchor_id], "Anchor: {:d} SIM is {}, ID-NEG: {:d} SIM is {}".format(anchor_id, sim_data[anchor_id], id_neg_id, sim_data[id_neg_id])
            assert anchor_id in dict_inv_ranking[id_neg_id]['same'], 'Anchor: {:d} is not in the "same" INV rank list of sample {:d}'.format(anchor_id, id_neg_id)
            anchor_rank = dict_inv_ranking[id_neg_id]['same'][anchor_id]
            axes[3, i].set_title(
                'In-dom. NEG Top-{:d}, rank-{:d}\n Dist: {:.1f} m, {:.1f} deg'.format(i + 1, anchor_rank + 1, t_err,
                                                                                      r_err))
    else:
        anchor_img = images[0].cpu().numpy().transpose(1, 2, 0)
        # show image
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(anchor_img)
        plt.title('Anchor ({:s}), infeasible.'.format('sim' if is_sim else 'real'))

    if anchor_id in feasible_anchor:
        if is_sim:
            plt.savefig(os.path.join(preview_dir, 'feasible_sim_anchor', 'anchor_{:d}.png'.format(anchor_id)),
                        bbox_inches='tight', pad_inches=0.1)
        else:
            plt.savefig(os.path.join(preview_dir, 'feasible_real_anchor', 'anchor_{:d}.png'.format(anchor_id)),
                        bbox_inches='tight', pad_inches=0.1)
    else:
        if is_sim:
            plt.savefig(os.path.join(preview_dir, 'infeasible_sim_anchor', 'anchor_{:d}.png'.format(anchor_id)),
                        bbox_inches='tight', pad_inches=0.1)
        else:
            plt.savefig(os.path.join(preview_dir, 'infeasible_real_anchor', 'anchor_{:d}.png'.format(anchor_id)),
                        bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main():
    """Main function."""

    """Initialization"""
    opt = config_parser()
    print(opt)

    top_n = 6
    trainset, trainset_supercon, trainset_loader = config_dataloader(opt.scene, top_n, opt.real_chunk)

    # read geo distance table (might be redundant but let's keep it for clarity)
    geo_dist_path = os.path.abspath(os.path.join(PROJECT_DIR, "datasets", opt.scene, 'geo_dist.dat'))
    geo_dist, sim_data, dict_name_to_idx, dict_idx_to_name, hyper_params, dict_supercon, feasible_anchor = load_geo_dist(geo_dist_path, retain_geo_dist=True)

    feasible_anchor = trainset_supercon.ext_feasible_anchor  # update trimmed anchors list

    # check the simple concatenation of sim-real data
    sim_size = sim_data.sum().item()
    real_size = len(trainset) - sim_size

    """Trim the raw data if some real data is omitted"""
    if opt.real_chunk < 1.0:
        # geo distance table
        rows_to_delete = []
        for row in range(len(geo_dist)):
            if trainset_supercon.dict_idx_table2loader[row] is None:
                rows_to_delete.append(row)
                geo_dist[row] = -1.0
                geo_dist[:, row] = -1.0
        assert (np.array(rows_to_delete) >= sim_size).all(), "Sim data order should not be altered!"
        geo_dist = geo_dist[geo_dist != -1.0]
        num_ele = np.sqrt(geo_dist.size(0))
        assert num_ele // 1 == num_ele and num_ele == len(trainset)
        num_ele = int(num_ele)
        geo_dist = geo_dist.reshape(num_ele, num_ele)  # [X, X] matrix

        # trim name to index mapping dictionary
        _dict_name_to_idx = dict_name_to_idx.copy()
        dict_name_to_idx = {}
        for idx_loader, rgb_file in enumerate(trainset_supercon.rgb_files):
            dict_name_to_idx[get_unique_file_name(rgb_file)] = idx_loader

    dict_ranking, dict_inv_ranking = compute_raw_ranking(geo_dist, sim_size)
    """Show preview one by one"""
    torch.cuda.empty_cache()

    preview_dir = _create_folder(opt.scene, opt.real_chunk)

    for idx, (images, gt_poses, gt_coords, focal_lengths, file_names) in enumerate(trainset_loader):
        print('Data instance no. {:d}...'.format(idx))
        file_names = np.asarray(file_names).reshape(-1)
        batch_identities = trainset_supercon.seq_batch_identities[idx]

        plot_preview(preview_dir, trainset, batch_identities, dict_inv_ranking, sim_data, dict_name_to_idx,
                     feasible_anchor, top_n, images, file_names)


if __name__ == '__main__':
    main()
