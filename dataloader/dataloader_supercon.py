"""
Dataloader implementation based on DSAC* code.  (modified)
https://github.com/vislearn/dsacstar/blob/master/dataset.py

Major contents of change:
- add custom batch sampler based on geometric similarity contrast

Copyright (c) 2020, Heidelberg University
Copyright (c) 2021, EPFL
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import pdb

import numpy as np
from typing import Union, Tuple
import torch
from torch.utils.data import Dataset

from utils import load_geo_dist, _safe_printout
from utils import get_unique_file_name
from .dataloader import CamLocDataset
from . import FLAG_ANCHOR, FLAG_POS_IN_DOM, FLAG_POS_CROSS_DOM, FLAG_NEG_IN_DOM, FLAG_NEG_CROSS_DOM


class SuperconBatchSampler(torch.utils.data.Sampler):
    def __init__(self, seq_batch_indices):
        self.seq_batch_indices = seq_batch_indices

    def __iter__(self):
        return iter(self.seq_batch_indices)

    def __len__(self):
        return len(self.seq_batch_indices)


class CamLocDatasetSupercon(CamLocDataset):
    """
    Camera localization dataset.
    Access to image, calibration and ground truth data given a dataset directory.
    Customized for supervised contrastive learning w/ geometric similarity.
    """

    def __init__(self, root_dir_sim,
                 root_dir_real,
                 mode=1,
                 sparse=True,
                 coord=True,
                 depth=False,
                 normal=False,
                 augment=False,
                 grayscale=False,
                 batch=True,
                 raw_image=False,
                 supercon=True,
                 sampling_pos_cross_dom_top_n=5,
                 sampling_pos_in_dom_top_n=5,
                 sampling_neg_cross_dom_top_n=5,
                 sampling_neg_in_dom_top_n=5,
                 retain_sim_sample=False,
                 retain_real_sample=True,
                 aug_rotation=30,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_contrast=0.1,
                 aug_brightness=0.1,
                 image_height=480,
                 real_chunk=None,
                 fullsize=False,
                 mute=False):
        """
        Constructor.
        Parameters:
            root_dir_sim:  Folder of the synthetic data.
            root_dir_real: Folder of the real data.
            mode:
                0 = RGB only, load no initialization targets,
                1 = RGB + ground truth scene coordinates, load or generate ground truth scene coordinate targets,
                2 = RGB-D, load camera coordinates instead of scene coordinates.
            sparse: for mode = 1 (RGB+GT SC), load sparse initialization targets when True, load dense depth maps and generate initialization targets when False.
            coord: Return 3D GT SC. (effecive iff mode == 1 && sparse == True)
            depth: Return z-buffer depth. (effecive iff mode == 1 && sparse == True)
            normal: Return surface normal. (effecive iff mode == 1 && sparse == True)
            # Note: if more than one labels shall be retrieved, the output is a dictionary; see the end of __getitem__ for details
            augment: Use random data augmentation, note: not supported for mode = 2 (RGB-D) since pre-generated eye coordinates cannot be augmented.
            grayscale: return single-channel gray scale RGB image.
            raw_image: Return raw RGB image w/o any augmentation or normalization for post-processing
            batch: Mini-batch mode (size > 1), affects the data augmentation during training
            supercon: enable the following supervised contrastive learning sampling dataloader:
            sampling_pos_cross_dom_top_n: for feasible anchor, select top_n (similar) cross-domain positives,
            sampling_pos_in_dom_top_n: for feasible anchor, select top_n (similar) in-domain positives,
            sampling_neg_cross_dom_top_n: for feasible anchor, select top_n (dissimilar) cross-domain negatives,
            sampling_neg_in_dom_top_n: for feasible anchor, select top_n (dissimilar) in-domain negatives.
            retain_sim_sample: to keep all synthetic instances (like infeasible anchor) in the dataloader.
            retain_real_sample: to keep all real instances (like infeasible anchor) in the dataloader.
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions.
            aug_scale_min: Lower limit of image scale factor for uniform sampling.
            aug_scale_min: Upper limit of image scale factor for uniform sampling.
            aug_contrast: Max relative scale factor for image contrast sampling, e.g. 0.1 -> [0.9,1.1].
            aug_brightness: Max relative scale factor for image brightness sampling, e.g. 0.1 -> [0.9,1.1].
            image_height: RGB images are rescaled to this maximum height.
            real_chunk: the proportion of real images to load.
            fullsize: to use full size 3D labels.
            mute: to print out I/O-related message.
        """

        root_dir_ls = [root_dir_sim, root_dir_real]
        self._config_dataloader(root_dir_ls, mode, sparse, coord, depth, normal, augment, grayscale, batch, raw_image,
                                aug_rotation, aug_scale_min, aug_scale_max, aug_contrast, aug_brightness, image_height,
                                real_chunk, fullsize, mute)
        # Parameters specific to contrastive learning
        self.supercon = supercon

        if self.supercon:
            geo_dist_path = os.path.abspath(os.path.join(root_dir_sim, '../geo_dist.dat'))
            self._init_supercon(geo_dist_path, sampling_pos_cross_dom_top_n, sampling_pos_in_dom_top_n,
                                sampling_neg_cross_dom_top_n, sampling_neg_in_dom_top_n,
                                retain_sim_sample, retain_real_sample)
        else:
            self.data_weight = np.ones(len(self.rgb_files))

    def _init_supercon(self, geo_dist_path, sampling_pos_cross_dom_top_n, sampling_pos_in_dom_top_n,
                       sampling_neg_cross_dom_top_n, sampling_neg_in_dom_top_n,
                       retain_sim_sample, retain_real_sample):
        """
        Initialize supervised contrastive learning based on 3D geometric similarity.
        """

        """Initialization"""
        _, sim_data, dict_name_to_idx, dict_idx_to_name, hyper_params, dict_supercon, _ = load_geo_dist(
            geo_dist_path, retain_geo_dist=False)
        sim_size = sim_data.sum().item()
        real_size = len(self.rgb_files) - sim_size

        self.sampling_pos_cross_dom_top_n = sampling_pos_cross_dom_top_n
        self.sampling_pos_in_dom_top_n = sampling_pos_in_dom_top_n
        self.sampling_neg_cross_dom_top_n = sampling_neg_cross_dom_top_n
        self.sampling_neg_in_dom_top_n = sampling_neg_in_dom_top_n
        self.len_batch = 1 + self.sampling_pos_cross_dom_top_n + self.sampling_pos_in_dom_top_n + self.sampling_neg_cross_dom_top_n + self.sampling_neg_in_dom_top_n
        self.retain_sim_sample = retain_sim_sample
        self.retain_real_sample = retain_real_sample

        assert self.sampling_pos_cross_dom_top_n, "#Cross-domain positives must be positive, now it's {:d}".format(self.sampling_pos_cross_dom_top_n)
        assert self.sampling_neg_cross_dom_top_n + self.sampling_neg_in_dom_top_n, "#Negatives in total must be positive, now it's {:d}".format(self.sampling_neg_cross_dom_top_n + self.sampling_neg_in_dom_top_n)

        """Anchors screening & augmenting"""
        # Check if the sampled data could be extended into an anchor w/ enough positives & negatives
        # The rule of thumb is to use more in-domain instances in case cross-domain instances are not enough
        ext_feasible_anchor, real_anchor, sim_anchor = [], [], []
        dataloader_dict = [None] * len(self.rgb_files)
        # Warning: # if real_chunk < 1.0, size of dataloader_dict is smaller than that of dict_supercon

        """Trim the data_supercon ranking list to adapt to real_chunk coefficient"""
        dict_idx_loader2table = {}
        dict_idx_table2loader = dict(zip(range(len(dict_supercon)), [None] * len(dict_supercon)))
        for idx_loader, rgb_file in enumerate(self.rgb_files):
            unique_file_name = get_unique_file_name(rgb_file).replace('_chunk_{:.2f}'.format(self.real_chunk), '')
            idx_table = dict_name_to_idx[unique_file_name]
            dict_idx_loader2table[idx_loader] = idx_table
            dict_idx_table2loader[idx_table] = idx_loader

        def _trim_index_ls(idx_ls, dict_idx_mapping):
            return [dict_idx_mapping[ele] for ele in idx_ls if dict_idx_mapping[ele] is not None]

        _dict_supercon = dict_supercon.copy()
        dict_supercon = [None] * len(self.rgb_files)
        for idx_table, idx_loader in zip(dict_idx_table2loader.keys(), dict_idx_table2loader.values()):
            if isinstance(idx_loader, int):
                trim_dict = _dict_supercon[idx_table].copy()
                for mode in ['positive', 'negative']:
                    for dom in ['diff', 'same']:
                        trim_dict[mode][dom] = _trim_index_ls(trim_dict[mode][dom], dict_idx_table2loader)
                        assert (np.array(trim_dict[mode][dom]) < len(self.rgb_files)).all()
                dict_supercon[idx_loader] = trim_dict
            else:
                pass

        for anchor_id, _ in enumerate(self.rgb_files):
            if self.sampling_pos_cross_dom_top_n > len(dict_supercon[anchor_id]['positive']['diff']):
                # Insufficient cross-domain positives
                continue
            if self.sampling_pos_in_dom_top_n > len(dict_supercon[anchor_id]['positive']['same']):
                # Insufficient in-domain positives
                continue
            if self.sampling_neg_cross_dom_top_n + self.sampling_neg_in_dom_top_n > len(
                    dict_supercon[anchor_id]['negative']['diff']) + len(dict_supercon[anchor_id]['negative']['same']):
                # Supplement in(cross) domain negatives with cross(in) domain negatives is allowed
                # This is disabled for positives for stronger contrastive strength
                continue
            dataloader_dict[anchor_id] = dict_supercon[anchor_id].copy()
            ext_feasible_anchor.append(anchor_id)
            is_anchor_sim = True if anchor_id < sim_size else False
            if is_anchor_sim:
                sim_anchor.append(anchor_id)
            else:
                real_anchor.append(anchor_id)

        _safe_printout('%d / %d samples are feasible anchors after anchor extension.' % (
            len(ext_feasible_anchor), len(self.rgb_files)))
        _safe_printout('Anchor distribution: {:d} / {:d} synthetic anchors, {:d} / {:d} real anchors.'.format(
            len(sim_anchor), sim_size, len(real_anchor), real_size))

        """Store data"""
        if self.real_chunk == 1.0 and len(self.rgb_files) != len(dict_supercon):
            raise Exception('Dataset size is not consistent with the geometric distance table!')

        self.ext_feasible_anchor = ext_feasible_anchor
        self.sim_anchor = sim_anchor
        self.real_anchor = real_anchor
        self.sim_size = sim_size
        self.dataloader_dict = dataloader_dict
        self.infeasible_anchor = np.setdiff1d(np.arange(len(self.rgb_files)), self.ext_feasible_anchor)
        self.dict_idx_loader2table = dict_idx_loader2table
        self.dict_idx_table2loader = dict_idx_table2loader

        # release GPU memory
        del sim_data, dict_name_to_idx, dict_idx_to_name, dict_supercon
        torch.cuda.empty_cache()

        # initialize the custom batch sampler
        self._shuffle_batch_sampler()

    def _shuffle_batch_sampler(self):
        """
        Shuffle the hard-coded batch sampler with a variable batch-size.
        It's advised to call this function per epoch for better dataset rollout.
        """
        seq_batch_indices = []
        seq_batch_identities = []
        data_freq = np.array([0] * len(self.rgb_files))  # cumulative visiting frequency for one epoch
        for anchor_id in range(len(self.rgb_files)):
            # skip some samples dependent on boundary conditions
            if anchor_id < self.sim_size:
                if not self.retain_sim_sample and anchor_id not in self.sim_anchor:
                    continue
            else:
                if not self.retain_real_sample and anchor_id not in self.real_anchor:
                    continue
            this_batch_indices = [anchor_id]
            this_batch_identities = [FLAG_ANCHOR]
            if anchor_id in self.ext_feasible_anchor:
                for mode in ['positive', 'negative']:
                    # Initialization
                    _flag_in_dom_sup = False
                    _flag_cross_dom_sup = False

                    if mode == 'positive':
                        _target_cross_dom = self.sampling_pos_cross_dom_top_n
                        _target_in_dom = self.sampling_pos_in_dom_top_n
                        _flag_cross_dom = FLAG_POS_CROSS_DOM
                        _flag_in_dom = FLAG_POS_IN_DOM
                    elif mode == 'negative':
                        _target_cross_dom = self.sampling_neg_cross_dom_top_n
                        _target_in_dom = self.sampling_neg_in_dom_top_n
                        _flag_cross_dom = FLAG_NEG_CROSS_DOM
                        _flag_in_dom = FLAG_NEG_IN_DOM
                    else:
                        raise NotImplementedError

                    _raw_in_dom_cand = self.dataloader_dict[anchor_id][mode]['same']
                    _raw_cross_dom_cand = self.dataloader_dict[anchor_id][mode]['diff']

                    if len(_raw_in_dom_cand) < _target_in_dom:
                        _flag_cross_dom_sup = True
                    if len(_raw_cross_dom_cand) < _target_cross_dom:
                        _flag_in_dom_sup = True
                    if _flag_cross_dom_sup and _flag_in_dom_sup:
                        raise Exception("Instance {:d} {:s} samples insufficient! Cross-dom={:d}, In-dom={:d}".format(
                            anchor_id, mode, len(_raw_cross_dom_cand), len(_raw_in_dom_cand)))

                    # Retrieve data in the first trial
                    for raw_cand, target, flag in zip([_raw_in_dom_cand, _raw_cross_dom_cand],
                                                      [_target_in_dom, _target_cross_dom],
                                                      [_flag_in_dom, _flag_cross_dom]):
                        if len(raw_cand) >= target:
                            this_batch_indices.extend(self._weighted_k_sample(raw_cand, target, data_freq))
                            this_batch_identities.extend([flag] * target)
                        else:
                            this_batch_indices.extend(np.random.permutation(raw_cand))
                            this_batch_identities.extend([flag] * len(raw_cand))

                    # Supplement the retrieved data in case of insufficient samples
                    num_samples = np.equal(this_batch_identities, _flag_in_dom).sum() + np.equal(this_batch_identities,
                                                                                                 _flag_cross_dom).sum()
                    num_samples_sup = int(_target_in_dom + _target_cross_dom - num_samples)
                    if num_samples_sup == 0:
                        pass
                    elif num_samples_sup > 0:
                        assert mode is not 'positive', "Positive samples retrieval should not use any supplementing!}"
                        if _flag_in_dom_sup:
                            this_batch_indices.extend(self._weighted_k_sample(_raw_in_dom_cand, num_samples_sup,
                                                                              data_freq, exclude=this_batch_indices))
                            this_batch_identities.extend([_flag_in_dom] * num_samples_sup)
                        if _flag_cross_dom_sup:
                            this_batch_indices.extend(self._weighted_k_sample(_raw_cross_dom_cand, num_samples_sup,
                                                                              data_freq, exclude=this_batch_indices))
                            this_batch_identities.extend([_flag_cross_dom] * num_samples_sup)
                    else:
                        raise NotImplementedError

            for this_idx in this_batch_indices:
                data_freq[this_idx] += 1
            seq_batch_indices.append(this_batch_indices)
            seq_batch_identities.append(this_batch_identities)

        # Sub-list order shuffle
        perm_idx = np.random.permutation(len(seq_batch_indices))
        seq_batch_indices = [seq_batch_indices[idx] for idx in perm_idx]
        seq_batch_identities = [seq_batch_identities[idx] for idx in perm_idx]

        # Sanity check to ensure there is no error
        for batch_indices, batch_identities in zip(seq_batch_indices, seq_batch_identities):
            if len(batch_indices) == 1:
                assert batch_identities == [FLAG_ANCHOR]
            elif len(batch_indices) == self.len_batch:
                assert batch_identities[0] == FLAG_ANCHOR
                pos_id = batch_identities[1:1+self.sampling_pos_in_dom_top_n+self.sampling_pos_cross_dom_top_n]
                assert np.array_equal(np.unique(pos_id).sort(), np.array([FLAG_POS_CROSS_DOM, FLAG_POS_IN_DOM]).sort())
                neg_id = batch_identities[1+self.sampling_pos_in_dom_top_n+self.sampling_pos_cross_dom_top_n:]
                assert np.array_equal(np.unique(neg_id).sort(), np.array([FLAG_NEG_CROSS_DOM, FLAG_NEG_IN_DOM]).sort())
            else:
                raise Exception("The length of current batch {:d} is wrong!".format(len(batch_indices)))

        # Compute weight for loss function
        # based on median frequency balancing as in the SegNet
        data_freq[data_freq > 0] = np.median(data_freq[data_freq > 0]) / data_freq[data_freq > 0]
        data_weight = data_freq * (data_freq > 0) + 0.0 * (data_freq <= 0)
        assert (data_weight >= 0).all() and (np.logical_not(np.isnan(data_weight))).all()

        self.seq_batch_indices = seq_batch_indices
        self.seq_batch_identities = seq_batch_identities
        self.data_freq = data_freq
        self.data_weight = data_weight

        _safe_printout("Restarted the custom sampler: {:d} iterations per epoch ({:.2f}X compared to vanilla "
                       "iteration). Ideal contrastive batch size is {:d}.".format(
                        data_freq.sum(), data_freq.sum() / len(self.rgb_files), self.len_batch))

    def get_supercon_sampler(self, shuffle=True):
        """
        Return pytorch dataloader-compatible batch sampler.
        @return:
        """
        if shuffle:
            self._shuffle_batch_sampler()
        else:
            pass
        return SuperconBatchSampler(self.seq_batch_indices)

    def get_aux_info(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return some helpful information for training.

        """
        batch_identities = self.seq_batch_identities[batch_idx]
        mask_anchor = self._get_id_mask(batch_identities, 'anchor')
        mask_positive = self._get_id_mask(batch_identities, 'positive')
        mask_negative = self._get_id_mask(batch_identities, 'negative')

        loss_weights = torch.tensor(self.data_weight[self.seq_batch_indices[batch_idx]])

        return mask_anchor, mask_positive, mask_negative, loss_weights

    @staticmethod
    def _weighted_k_sample(sample_ls: np.ndarray, k: int, sample_freq: np.ndarray,
                           exclude: Union[np.ndarray, list, None] = None) -> np.ndarray:
        """
        Sample k instances according to the weight (dependent on the data frequency).
        @param sample_ls:       a list for the candidate samples' indices
        @param k:               the number of instances to sample
        @param sample_freq:     a list recording the cumulative frequency for all instances
        @param exclude:         excluding some candidate samples
        @return:                indices for the selected k samples
        """
        if exclude is not None:
            exclude = np.array(exclude) if not isinstance(exclude, np.ndarray) else exclude
            sample_ls = np.setdiff1d(sample_ls.copy(), exclude)
        weight = -(sample_freq[sample_ls] - np.min(sample_freq[sample_ls]) + 1)
        weight = weight / np.sum(weight)
        selected_idx = np.random.choice(sample_ls, k, replace=False, p=weight)
        return selected_idx

    @staticmethod
    def _get_id_mask(identities: list, mode: str) -> torch.Tensor:
        """
        Return masks for anchor/positive/negative.
        @param identities:  a list size of B, consisting of the defined flags.
        @param mode:        search for either 'anchor', 'positive' or 'negative' data instances.
        @return:            a boolean mask size of B, True for the searched data instance.
        """
        if mode == 'anchor':
            flag_targets = [FLAG_ANCHOR]
            pass
        elif mode == 'positive':
            flag_targets = [FLAG_POS_IN_DOM, FLAG_POS_CROSS_DOM]
        elif mode == 'negative':
            flag_targets = [FLAG_NEG_IN_DOM, FLAG_NEG_CROSS_DOM]
        else:
            raise NotImplementedError

        masks = np.zeros(len(identities)).astype(bool)
        for flag_target in flag_targets:
            masks = np.logical_or(masks, np.equal(identities, flag_target))
        return torch.tensor(masks)

