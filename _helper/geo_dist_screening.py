import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pdb
import pickle
import multiprocessing as mp

from tqdm import tqdm

from IPython.display import clear_output

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, PROJECT_DIR)

# from visualize_geo_dist import load_geo_dist
from dataloader.dataloader import CamLocDataset
from dataloader.dataloader_supercon import CamLocDatasetSupercon
from utils import get_unique_file_name
from datasets.setup_geo_distance import compute_raw_ranking


def load_geo_dist(scene):
    """
    Load pre-computed geometric distance table.
    """
    geo_dist_path = os.path.abspath(os.path.join(PROJECT_DIR, "datasets", scene, 'geo_dist.dat'))
    assert os.path.exists(geo_dist_path)
    print('Load geometric distance from {:s}...'.format(geo_dist_path))
    geo_dist_dict = torch.load(geo_dist_path)
    geo_dist = geo_dist_dict['geo_dist'].cpu()  # raw distance table, debug purpose

    sim_data = geo_dist_dict['sim_data'].cpu()
    dict_name_to_idx = geo_dist_dict['dict_name_to_idx']
    dict_idx_to_name = geo_dist_dict['dict_idx_to_name']
    hyper_params = geo_dist_dict['hyper_params']
    dict_supercon = geo_dist_dict['dict_supercon']
    dict_ranking = geo_dist_dict['dict_ranking']  # raw ranking dict, debug purpose
    feasible_anchor = geo_dist_dict['feasible_anchor']
    trans_dist = geo_dist_dict['trans_dist'].cpu()

    # dict_inv_ranking_path = os.path.abspath(os.path.join(PROJECT_DIR, "datasets", scene, 'dict_inv_ranking.dat'))
    # with open(dict_inv_ranking_path, 'rb') as f:
    #     dict_inv_ranking = pickle.load(f)

    _, dict_inv_ranking = compute_raw_ranking(geo_dist, torch.sum(sim_data).item())

    # pdb.set_trace()

    return sim_data, dict_name_to_idx, dict_idx_to_name, hyper_params, dict_supercon, dict_ranking, dict_inv_ranking, feasible_anchor, trans_dist


def check_backward_ranking(sample_ranking, anchor_id, k_percentage, reverse=False):
    """
    Check backward ranking/re-ranking constraint.
    """
    if isinstance(sample_ranking, list) and isinstance(sample_ranking[0], np.ndarray):
        sample_ranking_cat = np.stack(sample_ranking, axis=0)  # [B, N]
        if reverse:
            sample_ranking_cat = sample_ranking_cat[:, ::-1]        
        b_rank_rows, backward_rank_ls = np.where(sample_ranking_cat == anchor_id)
        assert np.array_equal(np.sort(b_rank_rows), np.arange(sample_ranking_cat.shape[0])), print(b_rank_rows)  # sanity check, anchor_id should appear in each row once and only once
        critical_backward_rank = np.ceil(sample_ranking_cat.shape[1] * k_percentage).astype('int')
        return backward_rank_ls < critical_backward_rank  # [B, ]
    elif isinstance(sample_ranking, np.ndarray):
        assert anchor_id in sample_ranking  # sanity check
        if reverse:
            sample_ranking = sample_ranking[::-1]
        backward_rank = int(np.where(sample_ranking == anchor_id)[0])  # reciprocal rank of anchor seen from sample
        critical_backward_rank = np.ceil(len(sample_ranking) * k_percentage).astype('int')  # top/bottom k% re-ranking
        return backward_rank < critical_backward_rank  # scalar
    else:
        raise NotImplementedError
            
            
def _backbone_trim_ranking(trans_dist_vec, dict_ranking,
                           in_dom_pos_cand, in_dom_pos_cand_inv_rank,
                           cross_dom_pos_cand, cross_dom_pos_cand_inv_rank,
                           in_dom_neg_cand, in_dom_neg_cand_inv_rank,
                           cross_dom_neg_cand, cross_dom_neg_cand_inv_rank,
                           pos_max_trans, pos_backward_coef, neg_backward_coef,
                           dict_supercon, i_anchor, feasible_anchor, mp_progress, mp_lock):

    print("\rProgress: {:d} / {:d}".format(mp_progress.value, len(dict_supercon)), flush=True, end=' ')
    # store top k% in-domain positives with re-ranking
    flag_re_ranking = in_dom_pos_cand_inv_rank < len(dict_ranking[in_dom_pos_cand[0]]['same']) * pos_backward_coef
    flag_pos_max_trans = np.take(trans_dist_vec, in_dom_pos_cand) < pos_max_trans
    in_dom_positives = in_dom_pos_cand[np.logical_and(flag_re_ranking, flag_pos_max_trans)]  # to save

    # store top k% cross-domain positives with re-ranking
    flag_re_ranking = cross_dom_pos_cand_inv_rank < len(dict_ranking[cross_dom_pos_cand[0]]['diff']) * pos_backward_coef
    flag_pos_max_trans = np.take(trans_dist_vec, cross_dom_pos_cand) < pos_max_trans
    cross_dom_positives = cross_dom_pos_cand[np.logical_and(flag_re_ranking, flag_pos_max_trans)]  # to save

    # store bottom k% in-domain negatives with re-ranking
    flag_re_ranking = in_dom_neg_cand_inv_rank > len(dict_ranking[in_dom_neg_cand[0]]['same']) * (1 - neg_backward_coef)
    in_dom_negatives = in_dom_neg_cand[flag_re_ranking]  # to save

    # store bottom k% cross-domain negatives with re-ranking
    flag_re_ranking = cross_dom_neg_cand_inv_rank > len(dict_ranking[cross_dom_neg_cand[0]]['diff']) * (1 - neg_backward_coef)
    cross_dom_negatives = cross_dom_neg_cand[flag_re_ranking]  # to save

    # aggregate into a dict
    dict_anchor = {
        "positive": {"same": in_dom_positives, "diff": cross_dom_positives},
        "negative": {"same": in_dom_negatives, "diff": cross_dom_negatives}
    }
    dict_supercon[i_anchor] = dict_anchor
    
    if len(cross_dom_positives) > 0:
        feasible_anchor.append(i_anchor)

    with mp_lock:
        mp_progress.value += 1
    print("\rProgress: {:d} / {:d}".format(mp_progress.value, len(dict_supercon)), flush=True, end=' ')
    

def trim_ranking(dict_ranking, dict_inv_ranking, trans_dist, pos_forward_coef, pos_backward_coef, pos_max_trans, neg_forward_coef, neg_backward_coef):
    """
    Thresholding to get feasible anchors and the associated prescreened positives and negatives.
    """
    
    dict_supercon = {}
    feasible_anchor = []

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

        if len(cross_dom_positives) > 0:
            feasible_anchor.append(i_anchor)

    return dict_supercon, feasible_anchor


def main():
    # Recap on raw data structure
    coords = torch.load(os.path.join(PROJECT_DIR, 'datasets/EPFL/test_real/init/EPFL_2020-09-17-piloted_00035_DJI_0067.dat'))
    print("Coordinate raw data shape {}".format(coords.shape))

    depth = torch.load(os.path.join(PROJECT_DIR, 'datasets/EPFL/test_real/depth/EPFL_2020-09-17-piloted_00035_DJI_0067.dat'))
    print("Depth raw data shape {}".format(depth.shape))

    normal = torch.load(os.path.join(PROJECT_DIR, 'datasets/EPFL/test_real/normal/EPFL_2020-09-17-piloted_00035_DJI_0067.dat'))
    print("Normal raw data shape {}".format(normal.shape))

    scene = 'EPFL'

    # read geo distance table
    sim_data, dict_name_to_idx, dict_idx_to_name, hyper_params, dict_supercon, dict_ranking, dict_inv_ranking, feasible_anchor, trans_dist = load_geo_dist(scene)
    print('%d / %d samples are feasible anchors after pre-screening.' % (len(feasible_anchor), len(sim_data)))

    # check the simple concatenation of sim-real data
    sim_size = sim_data.sum().item()
    for i in range(len(sim_data)):
        if i < sim_size:
            assert sim_data[i]
        else:
            assert ~sim_data[i]
    real_size = len(sim_data) - sim_size

    torch.cuda.empty_cache()

    trans_dist = trans_dist.cpu().numpy()
    new_dict_supercon, new_feasible_anchor = trim_ranking(dict_ranking, dict_inv_ranking, trans_dist, pos_max_trans=150,
                                                          pos_forward_coef=0.01, pos_backward_coef=0.05,
                                                          neg_forward_coef=0.05, neg_backward_coef=0.05)

    pdb.set_trace()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    main()
