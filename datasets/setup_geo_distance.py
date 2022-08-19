import argparse
import os
import pdb
import sys
import time
import numpy as np

from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset

mp.set_sharing_strategy('file_system')

PROJECT_DIR = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, PROJECT_DIR)
from dataloader.dataloader import CamLocDataset
from utils import get_nodata_value, pick_valid_points, get_unique_file_name


def config_parser():
    """
    Config parser.
    """
    parser = argparse.ArgumentParser(
        description='Compute robust geometric distance between any two datapoints',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('scene', help='name of a scene in the dataset folder')
    parser.add_argument('--weight_pc', type=float, default=0.5, help='geo distance weight of 3D point cloud overlap')
    parser.add_argument('--weight_rot', type=float, default=0.5, help='geo distance weight of rotational difference')
    parser.add_argument('--weight_trans', type=float, default=0.0, help='geo distance weight of translational difference')

    parser.add_argument('--sat_rot', type=float, default=90.0, help='rotational difference saturation value')
    parser.add_argument('--pos_forward_coef', type=float, default=0.01,
                        help='select top k%% data for positive samples in the forward ranking loop')
    parser.add_argument('--pos_backward_coef', type=float, default=0.05,
                        help='select top k%% data for positive samples in the backward ranking / re-ranking loop')
    parser.add_argument('--pos_max_trans', type=float, default=150.0,
                        help='maximum translational distance for positive images')
    parser.add_argument('--neg_forward_coef', type=float, default=0.05,
                        help='select bottom n%% data for negatives in the forward ranking loop')
    parser.add_argument('--neg_backward_coef', type=float, default=0.05,
                        help='select bottom n%% data for negatives in the backward ranking / re-ranking loop')

    opt = parser.parse_args()
    if 'comballaz' in opt.scene:
        opt.pos_max_trans = 300.0
    elif 'EPFL' == opt.scene:
        opt.pos_max_trans = 150.0
    else:
        raise NotImplementedError
    return opt


def get_geo_dist(pc_i, pc_js, trans_mat_i, trans_mat_js, valid_data_i, valid_data_js, rot_dist, trans_dist,
                 weight_pc, weight_rot, weight_trans):
    """
    Get geometric distance between two point clouds
    @param pc_i           [4, N] tensor for homogeneous scene coordinates.
    @param pc_js          [B, 4, N] tensor for homogeneous scene coordinates.
    @param trans_mat_i    [3, 4] tensor for composite transformation.
    @param trans_mat_js   [B, 3, 4] tensor for composite transformation.
    @param valid_data_i   [N,  ] tensor for valid data indicator.
    @param valid_data_js  [B, N] tensor for valid data indicator.
    @param rot_dist       [B,  ], rotational distance between two poses, range [0, 1].
    @param trans_dist     [B,  ], translational distance between two poses.
    @param weight_pc      scalar, weight of 3D point cloud overlap.
    @param weight_rot     scalar, weight of rotational difference.
    @param weight_trans   scalar, weight of translational difference.
    @return geo_dist      [B,  ], 3D geometric distance between two point clouds & camera poses, range [0, 100].
    """
    repro_i_to_j = torch.matmul(trans_mat_js, pc_i)  # [B, 3, N]
    repro_j_to_i = torch.matmul(trans_mat_i.unsqueeze(0), pc_js)  # [B, 3, N]

    valid_i, valid_j = torch.stack([valid_data_i.clone()] * pc_js.size(0), dim=0), valid_data_js.clone()  # [B, N]
    valid_i &= (repro_i_to_j[:, -1, :] > 0.1)  # minimum depth requirement, [B, N]
    valid_j &= (repro_j_to_i[:, -1, :] > 0.1)  # [B, N]

    # restore to homogeneous pixel coordinates
    repro_i_to_j[:, 2, :].clamp_(min=0.1)  # avoid division by zero
    repro_i_to_j = repro_i_to_j[:, 0:2, :] / repro_i_to_j[:, 2, :].unsqueeze(1)  # [B, 2, N]

    repro_j_to_i[:, 2, :].clamp_(min=0.1)  # avoid division by zero
    repro_j_to_i = repro_j_to_i[:, 0:2, :] / repro_j_to_i[:, 2, :].unsqueeze(1)  # [B, 2, N]

    # check inlier rate (roughly visible in another camera pose)
    x_min, x_max, y_min, y_max = 0, 720, 0, 480

    inlier_i_to_j = (repro_i_to_j[:, 0] > x_min) & (repro_i_to_j[:, 0] < x_max) & (repro_i_to_j[:, 1] > y_min) & (
            repro_i_to_j[:, 1] < y_max)  # [B, N]
    inlier_i_to_j &= valid_i  # [B, N]
    inlier_i_to_j = inlier_i_to_j.sum(dim=-1) / float(pc_i.size(1))  # [B, ]
    inlier_j_to_i = (repro_j_to_i[:, 0] > x_min) & (repro_j_to_i[:, 0] < x_max) & (repro_j_to_i[:, 1] > y_min) & (
            repro_j_to_i[:, 1] < y_max)
    inlier_j_to_i &= valid_j  # [B, N]
    inlier_j_to_i = inlier_j_to_i.sum(dim=-1) / float(pc_i.size(1))  # [B, ]
    mutual_repro_rate = 0.5 * (abs(1 - inlier_i_to_j) + abs(1 - inlier_j_to_i))  # [B, ], range [0, 1]

    # aggregated geometric loss
    geo_dist = weight_pc * mutual_repro_rate + weight_rot * rot_dist + weight_trans * trans_dist  # range [0, 1.0]
    geo_dist = geo_dist * 100.0 / (weight_pc + weight_rot + weight_trans)  # [0, 100]

    return geo_dist


def config_dataloader(scene):
    """
    Config dataloader for original dataset w/o augmentation, used for initialization.
    """
    trainset_sim = CamLocDataset(os.path.join(PROJECT_DIR, 'datasets', scene, 'train_sim_aug'), mode=1,
                                 sparse=True, augment=False, grayscale=False)
    trainset_real = CamLocDataset(os.path.join(PROJECT_DIR, 'datasets', scene, 'train_real'), mode=1,
                                  sparse=True, augment=False, grayscale=False)
    trainset = ConcatDataset([trainset_sim, trainset_real])
    trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=False, num_workers=int(mp.cpu_count()))

    return trainset, trainset_loader, trainset_sim, trainset_real


def config_hp_dict(opt):
    """
    Get hyper-parameter dict for debug purpose.
    """
    if 'comballaz' in opt.scene or 'EPFL' == opt.scene:
        pass
    else:
        raise NotImplementedError
    hyper_params = {"weight_pc": opt.weight_pc,
                    "weight_rot": opt.weight_rot,
                    "weight_trans": opt.weight_trans,
                    "sat_rot": opt.sat_rot,
                    "pos_forward_coef": opt.pos_forward_coef,
                    "pos_backward_coef": opt.pos_backward_coef,
                    "pos_max_trans": opt.pos_max_trans,
                    "neg_forward_coef": opt.neg_forward_coef,
                    "neg_backward_coef": opt.neg_backward_coef
                    }
    return hyper_params


def data_preparation(trainset_loader, nodata_value, dict_name_to_idx, dict_idx_to_name):
    """
    First loop, extract easily-computed info. for later use.
    """
    # intermediate results
    point_clouds = [None] * len(trainset_loader)
    transform_mat = [None] * len(trainset_loader)
    valid_data = [None] * len(trainset_loader)
    cam_poses = [None] * len(trainset_loader)

    # create camera calibration matrix & auxiliary ones tensor only once
    for i, (image, gt_pose, gt_coords, focal_length, file) in enumerate(
            tqdm(trainset_loader, desc='Extract once-and-for-all information: O(n)')):
        if i == 0:
            focal_length = float(focal_length[0])
            cam_mat = torch.eye(3)
            cam_mat[0, 0] = focal_length
            cam_mat[1, 1] = focal_length
            cam_mat[0, 2] = image.size(3) / 2
            cam_mat[1, 2] = image.size(2) / 2
            cam_mat = cam_mat  # [3, 3]

            ones_ = torch.ones([1, gt_coords.size(-1) * gt_coords.size(-2)], dtype=torch.float32)  # [1, N]

        unique_file_nm = get_unique_file_name(file[0])
        dict_name_to_idx[unique_file_nm] = i
        dict_idx_to_name[i] = unique_file_nm

        point_clouds[i] = torch.cat([gt_coords.view(3, -1), ones_], dim=0)  # [4, N]
        transform_mat[i] = torch.mm(cam_mat, torch.inverse(gt_pose.view(4, 4))[0:3, :])  # [3, 3] * [3, 4]  --> [3, 4]
        valid_data[i] = pick_valid_points(gt_coords.view(3, -1), nodata_value, boolean=True)  # [H_ds*W_ds]
        cam_poses[i] = gt_pose.view(4, 4)

    # change into tensor for CUDA acceleration
    point_clouds = torch.stack(point_clouds, dim=0).cuda()
    transform_mat = torch.stack(transform_mat, dim=0).cuda()
    valid_data = torch.stack(valid_data, dim=0).cuda()
    cam_poses = torch.stack(cam_poses, dim=0).cuda()

    return point_clouds, transform_mat, valid_data, cam_poses, dict_name_to_idx, dict_idx_to_name


def compute_rot_trans_dist(trainset_loader, cam_poses):
    """
    Second loop, O(n) computation for rotational and translational distances
    """
    # intermediate variables
    rot_dist = torch.zeros(len(trainset_loader), len(trainset_loader), dtype=torch.float32).cuda()
    trans_dist = torch.zeros(len(trainset_loader), len(trainset_loader), dtype=torch.float32).cuda()

    pose_rots = cam_poses[:, :3, :3]  # [N, 3, 3]
    pose_trans = cam_poses[:, :3, -1]  # [N, 3]
    for i in tqdm(range(len(trainset_loader)), desc='Compute rotational & translational distance: O(n)'):
        pose_i_t = cam_poses[i][0:3, 0:3].t().unsqueeze(0)  # [1, 3, 3]
        rot_err = torch.matmul(pose_i_t, pose_rots)  # [N, 3, 3]
        batch_trace = rot_err.diagonal(offset=0, dim1=1, dim2=2).sum(-1)  # [N, ]
        safe_acos_input = torch.clamp((batch_trace - 1) / 2, min=-0.999999, max=0.999999)  # [N, ]
        rot_dist[i] = torch.acos(safe_acos_input) / np.pi  # [N, ], range [0, 1]

        trans_dist[i] = torch.norm(cam_poses[i][0:3, 3] - pose_trans, p=2, dim=-1)  # [N, ]

    return rot_dist, trans_dist


def compute_geo_dist_table(geo_dist, point_clouds, transform_mat, valid_data, rot_dist, trans_dist, weight_pc, weight_rot, weight_trans):
    """
    Third loop, O(n) computation for geometric distances.
    """
    for i in tqdm(range(len(geo_dist)), desc='Compute geometric distance: O(n)'):
        torch.cuda.empty_cache()
        if i < len(geo_dist) - 1:
            geo_dist_out = get_geo_dist(point_clouds[i], point_clouds[i:], transform_mat[i], transform_mat[i:],
                                        valid_data[i], valid_data[i:], rot_dist[i, i:], trans_dist[i, i:],
                                        weight_pc, weight_rot, weight_trans)
        else:
            geo_dist_out = get_geo_dist(point_clouds[i], point_clouds[i:i+1], transform_mat[i], transform_mat[i:i+1],
                                        valid_data[i], valid_data[i:i+1], rot_dist[i, i:i+1], trans_dist[i, i:i+1],
                                        weight_pc, weight_rot, weight_trans)
        geo_dist[i, i:] = geo_dist_out
        geo_dist[i:, i] = geo_dist_out
    return geo_dist


def compute_raw_ranking(geo_dist, sim_size):
    """
    Compute raw ranking table.
    """
    # obtain raw ranking by domain
    dict_ranking = [None] * len(geo_dist)  # intermediate variable, forward ranking, i.e., rank[order] = index, return the sample index for the rank
    dict_inv_ranking = [None] * len(geo_dist)  # inverse of the permutation, i.e., rank[index] = order, return the rank for the sample index
    for i, dist in enumerate(tqdm(geo_dist, desc='Computing ranking dict: O(n)')):
        idx_argsort_ = dist.argsort().cpu().numpy()
        idx_argsort_sim = idx_argsort_[idx_argsort_ < sim_size]
        idx_argsort_real = idx_argsort_[idx_argsort_ >= sim_size]

        idx_argsort_sim = np.delete(idx_argsort_sim, np.where(idx_argsort_sim == i)[0])
        idx_argsort_real = np.delete(idx_argsort_real, np.where(idx_argsort_real == i)[0])

        assert i not in idx_argsort_sim
        assert i not in idx_argsort_real

        # rank of the sample index
        inv_idx_argsort_ = np.argsort(idx_argsort_)
        subdict_inv_ranking = {'same': {}, 'diff': {}}
        for i_sample, inv_rank in enumerate(inv_idx_argsort_):
            if i == i_sample:
                continue
            if (i < sim_size and i_sample < sim_size) or (i >= sim_size and i_sample >= sim_size):
                subdict_inv_ranking['same'][i_sample] = inv_rank
            else:
                subdict_inv_ranking['diff'][i_sample] = inv_rank

        if i < sim_size:  # anchor id is synthetic data
            dict_ranking[i] = {'same': idx_argsort_sim, 'diff': idx_argsort_real}
        else:  # anchor id is real data
            dict_ranking[i] = {'same': idx_argsort_real, 'diff': idx_argsort_sim}

        dict_inv_ranking[i] = subdict_inv_ranking

    return dict_ranking, dict_inv_ranking


def trim_ranking(dict_ranking, dict_inv_ranking, trans_dist,
                 pos_forward_coef, pos_backward_coef, pos_max_trans, neg_forward_coef, neg_backward_coef):
    """
    Thresholding to get feasible anchors and the associated prescreened positives and negatives.
    """
    dict_supercon = [None] * len(dict_ranking)
    feasible_anchor = []
    trans_dist = trans_dist.cpu().numpy()

    for i_anchor in tqdm(range(len(dict_ranking)), desc='Sample thresholding based on reciprocal rank: O(n)'):
        trans_dist_vec = trans_dist[i_anchor]

        # store top k% in-domain positives with re-ranking
        in_dom_pos_cand = dict_ranking[i_anchor]['same']
        in_dom_pos_cand = in_dom_pos_cand[:np.ceil(pos_forward_coef * len(in_dom_pos_cand)).astype('int')]
        in_dom_pos_cand_inv_rank = np.array([dict_inv_ranking[i_sample]['same'][i_anchor] for i_sample in in_dom_pos_cand])
        flag_re_ranking = in_dom_pos_cand_inv_rank < len(dict_ranking[in_dom_pos_cand[0]]['same']) * pos_backward_coef
        flag_pos_max_trans = np.take(trans_dist_vec, in_dom_pos_cand) < pos_max_trans
        in_dom_positives = in_dom_pos_cand[np.logical_and(flag_re_ranking, flag_pos_max_trans)]  # to save

        # store top k% cross-domain positives with re-ranking
        cross_dom_pos_cand = dict_ranking[i_anchor]['diff']
        cross_dom_pos_cand = cross_dom_pos_cand[:np.ceil(pos_forward_coef * len(cross_dom_pos_cand)).astype('int')]
        cross_dom_pos_cand_inv_rank = np.array([dict_inv_ranking[i_sample]['diff'][i_anchor] for i_sample in cross_dom_pos_cand])
        flag_re_ranking = cross_dom_pos_cand_inv_rank < len(dict_ranking[cross_dom_pos_cand[0]]['diff']) * pos_backward_coef
        flag_pos_max_trans = np.take(trans_dist_vec, cross_dom_pos_cand) < pos_max_trans
        cross_dom_positives = cross_dom_pos_cand[np.logical_and(flag_re_ranking, flag_pos_max_trans)]  # to save

        # store bottom k% in-domain negatives with re-ranking
        in_dom_neg_cand = dict_ranking[i_anchor]['same']
        in_dom_neg_cand = in_dom_neg_cand[-np.ceil(neg_forward_coef * len(in_dom_neg_cand)).astype('int'):]
        in_dom_neg_cand_inv_rank = np.array([dict_inv_ranking[i_sample]['same'][i_anchor] for i_sample in in_dom_neg_cand])
        flag_re_ranking = in_dom_neg_cand_inv_rank > len(dict_ranking[in_dom_neg_cand[0]]['same']) * (1 - neg_backward_coef)
        in_dom_negatives = in_dom_neg_cand[flag_re_ranking]  # to save

        # store bottom k% cross-domain negatives with re-ranking
        cross_dom_neg_cand = dict_ranking[i_anchor]['diff']
        cross_dom_neg_cand = cross_dom_neg_cand[-np.ceil(neg_forward_coef * len(cross_dom_neg_cand)).astype('int'):]
        cross_dom_neg_cand_inv_rank = np.array([dict_inv_ranking[i_sample]['diff'][i_anchor] for i_sample in cross_dom_neg_cand])
        flag_re_ranking = cross_dom_neg_cand_inv_rank > len(dict_ranking[cross_dom_neg_cand[0]]['diff']) * (1 - neg_backward_coef)
        cross_dom_negatives = cross_dom_neg_cand[flag_re_ranking]  # to save

        # aggregate into a dict
        dict_anchor = {
            "positive": {"same": in_dom_positives, "diff": cross_dom_positives},
            "negative": {"same": in_dom_negatives, "diff": cross_dom_negatives}
        }
        dict_supercon[i_anchor] = dict_anchor

        if len(in_dom_positives) and len(cross_dom_positives) and len(in_dom_negatives) and len(cross_dom_negatives):
            feasible_anchor.append(i_anchor)

    return dict_supercon, feasible_anchor


def main():
    """
    Calculate geometric distance between any two data points.
    """

    """Initialization"""
    total_time = time.time()

    opt = config_parser()
    print(opt)

    trainset, trainset_loader, trainset_sim, _ = config_dataloader(opt.scene)
    print("Found %d training images for %s." % (len(trainset), opt.scene))

    print("Preparing geometrical distance table for the scene...")

    nodata_value = get_nodata_value(opt.scene)

    geo_dist_path = os.path.abspath(os.path.join(PROJECT_DIR, 'datasets', opt.scene, 'geo_dist.dat'))

    """Important variables/parameters of interest"""
    # variables to save
    geo_dist = torch.zeros([len(trainset), len(trainset)], dtype=torch.float32)
    sim_data = torch.zeros(len(trainset)).bool()
    dict_name_to_idx, dict_idx_to_name = {}, {}

    sim_data[0:len(trainset_sim)] = True
    sim_size = sim_data.sum().item()

    # hyper-parameters
    hyper_params = config_hp_dict(opt)
    weight_pc, weight_rot, weight_trans = hyper_params['weight_pc'], hyper_params['weight_rot'], hyper_params['weight_trans']
    sat_rot = hyper_params['sat_rot']
    pos_forward_coef, pos_backward_coef = hyper_params['pos_forward_coef'], hyper_params['pos_backward_coef']
    pos_max_trans = hyper_params['pos_max_trans']
    neg_forward_coef, neg_backward_coef = hyper_params['neg_forward_coef'], hyper_params['neg_backward_coef']  # bottom k% data for negative samples

    """ First loop, extract easily-computed info. """
    point_clouds, transform_mat, valid_data, cam_poses, dict_name_to_idx, dict_idx_to_name = data_preparation(
        trainset_loader, nodata_value, dict_name_to_idx, dict_idx_to_name)

    """ Second loop, O(n) computation for rotational and translational distances"""
    rot_dist, trans_dist = compute_rot_trans_dist(trainset_loader, cam_poses)
    rot_dist.clamp_(max=sat_rot / 180.0)  # saturation for rotational distance
    rot_dist = rot_dist / rot_dist.cpu().max()  # range [0, 1]

    # release GPU memory
    del cam_poses
    torch.cuda.empty_cache()

    """ Third loop, O(n) computation for geometric distances """
    geo_dist = compute_geo_dist_table(geo_dist, point_clouds, transform_mat, valid_data, rot_dist, trans_dist, weight_pc, weight_rot, weight_trans)

    """Some post-processing"""
    # obtain raw ranking w.r.t. domain
    dict_ranking, _dict_inv_ranking = compute_raw_ranking(geo_dist, sim_size)

    # trim the raw ranking results
    dict_supercon, feasible_anchor = trim_ranking(dict_ranking, _dict_inv_ranking, trans_dist,
                                                  pos_forward_coef, pos_backward_coef, pos_max_trans, neg_forward_coef, neg_backward_coef)

    print('%d / %d samples are feasible anchors.' % (len(feasible_anchor), len(geo_dist)))

    """Save data"""
    geo_dist_dict = {'geo_dist': geo_dist.cpu(), 'sim_data': sim_data.cpu(),
                     'dict_name_to_idx': dict_name_to_idx, 'dict_idx_to_name': dict_idx_to_name,
                     'hyper_params': hyper_params,
                     'dict_supercon': dict_supercon, 'feasible_anchor': np.array(feasible_anchor),
                     'trans_dist': trans_dist.cpu()
                     }
    torch.save(geo_dist_dict, geo_dist_path)
    print('Save geometric distance to {:s}...'.format(geo_dist_path))

    total_time = time.time() - total_time
    print('Total time: {:.1f} s'.format(total_time))


if __name__ == '__main__':
    main()
