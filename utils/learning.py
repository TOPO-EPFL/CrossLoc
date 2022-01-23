import copy
import logging
import math
import os
import pdb
import random

import cv2
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

import torch
from torch import multiprocessing as mp, optim as optim, nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from dataloader.dataloader import CamLocDataset
from networks.networks import TransPoseNet, Network


def get_pixel_grid(SUBSAMPLE):
    """
    Generate grid of target reprojection pixel positions (tensor)
    """
    pixel_grid = torch.zeros((2,
                              math.ceil(1080 / SUBSAMPLE),
                              # 1200px is max limit of image size, increase if needed
                              math.ceil(1080 / SUBSAMPLE)))

    for x in range(0, pixel_grid.size(2)):
        for y in range(0, pixel_grid.size(1)):
            pixel_grid[0, y, x] = x * SUBSAMPLE + SUBSAMPLE / 2
            pixel_grid[1, y, x] = y * SUBSAMPLE + SUBSAMPLE / 2

    pixel_grid = pixel_grid.cuda()
    return pixel_grid


def get_nodata_value(scene_name):
    """
    Get nodata value based on dataset scene name.
    """
    if 'urbanscape' in scene_name.lower() or 'naturescape' in scene_name.lower():
        nodata_value = -1
    else:
        raise NotImplementedError
    return nodata_value


def pick_valid_points(coord_input, nodata_value, boolean=False):
    """
    Pick valid 3d points from provided ground-truth labels.
    @param   coord_input   [B, C, N] or [C, N] tensor for 3D labels such as scene coordinates or depth.
    @param   nodata_value  Scalar to indicate NODATA element of ground truth 3D labels.
    @param   boolean       Return boolean variable or explicit index.
    @return  val_points    [B, N] or [N, ] Boolean tensor or valid points index.
    """
    batch_mode = True
    if len(coord_input.shape) == 2:
        # coord_input shape is [C, N], let's make it compatible
        batch_mode = False
        coord_input = coord_input.unsqueeze(0)  # [B, C, N], with B = 1

    val_points = torch.sum(coord_input == nodata_value, dim=1) == 0  # [B, N]
    val_points = val_points.to(coord_input.device)
    if not batch_mode:
        val_points = val_points.squeeze(0)  # [N, ]
    if boolean:
        pass
    else:
        val_points = torch.nonzero(val_points, as_tuple=True)  # a tuple for rows and columns indices
    return val_points


def set_random_seed(random_seed):
    """
    Set random seeds for reproducibility
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def get_label_mean(trainset_loader, nodata_value, scene, task):
    """
    Calculate mean of ground truth scene coordinates.
    """
    if task == 'coord':
        mean = torch.zeros(3)
        count = 0
        if 'naturescape' in scene:
            mean = torch.tensor([-455.934, 417.50, 520.31]).float()
        elif 'urbanscape' in scene:
            mean = torch.tensor([-29.34, 184.17, 91.96]).float()
        else:
            for idx, (images, gt_poses, gt_coords, focal_lengths, file_names) in enumerate(trainset_loader):
                # batch-size must be one for compatability
                # use mean of ground truth scene coordinates
                if isinstance(gt_coords, dict):
                    gt_coords = gt_coords['coord']
                elif isinstance(gt_coords, torch.Tensor):
                    pass
                else:
                    raise NotImplementedError
                gt_coords = gt_coords[0].view(3, -1)
                coord_mask = pick_valid_points(gt_coords.view(3, -1), nodata_value, boolean=True)  # [H_ds*W_ds]
                if coord_mask.sum() > 0:
                    gt_coords = gt_coords[:, coord_mask]
                    mean += gt_coords.sum(1)
                    count += coord_mask.sum()
            mean /= count
    elif task == 'depth':
        mean = torch.zeros(1)
        count = 0
        if 'naturescape' in scene:
            mean = torch.tensor([241.47]).float()
        elif 'urbanscape' in scene:
            mean = torch.tensor([136.24]).float()
        else:
            for idx, (images, gt_poses, gt_depths, focal_lengths, file_names) in enumerate(trainset_loader):
                # batch-size must be one for compatability
                # use mean of ground truth depth
                if isinstance(gt_depths, dict):
                    gt_depths = gt_depths['depth']
                elif isinstance(gt_depths, torch.Tensor):
                    pass
                else:
                    raise NotImplementedError
                gt_depths = gt_depths[0].view(1, -1)  # [1, H_ds * W_ds]
                depth_mask = pick_valid_points(gt_depths.view(1, -1), nodata_value, boolean=True)  # [H_ds*W_ds]
                if depth_mask.sum() > 0:
                    gt_depths = gt_depths[:, depth_mask]
                    mean += gt_depths.sum(1)
                    count += depth_mask.sum()
            mean /= count
    elif task == 'normal':
        mean = torch.zeros(2)
        count = 0

        def _inverse_sigmoid(logit):
            return -torch.log((1 / (logit + 1.e-7)) - 1.0)

        if 'naturescape' in scene:
            mean = (torch.tensor([-0.7943, -0.9986]) / np.pi + 1.0) / 2.0  # angles to raw sigmoid output
            mean = _inverse_sigmoid(mean).float()
        elif 'urbanscape' in scene:
            mean = (torch.tensor([-1.0454, -0.9858]) / np.pi + 1.0) / 2.0  # angles to raw sigmoid output
            mean = _inverse_sigmoid(mean).float()
        else:
            for idx, (images, gt_poses, gt_normals, focal_lengths, file_names) in enumerate(trainset_loader):
                # batch-size must be one for compatability
                # use mean of ground truth depth
                print('idx=', idx)
                if isinstance(gt_normals, dict):
                    gt_normals = gt_normals['normal']
                elif isinstance(gt_normals, torch.Tensor):
                    pass
                else:
                    raise NotImplementedError
                gt_normals = gt_normals.view(1, 3, -1)  # [1, 3, H_ds * W_ds]
                gt_normals_ae = xyz2ae(gt_normals).view(2, -1)  # [2, H_ds * W_ds]
                normal_mask = pick_valid_points(gt_normals.view(3, -1), nodata_value, boolean=True)  # [H_ds*W_ds]
                if normal_mask.sum() > 0:
                    gt_normals_ae = gt_normals_ae[:, normal_mask]  # [2, N]
                    mean += gt_normals_ae.sum(1)
                    count += normal_mask.sum()
            mean /= count
            mean = torch.atanh(mean / np.pi)  # logits before tanh <- true angle in radian
    elif task == 'semantics':
        mean = torch.zeros(6)
    else:
        raise NotImplementedError

    return mean


def config_dataloader(scene, task, grayscale, real_data_domain, real_data_chunk, sim_data_chunk,
                      fullsize, batch_size, nodata_value, real_only=False):
    """
    Configure dataloader (task specific).
    """

    # sanity check
    if 'urbanscape' in scene.lower() or 'naturescape' in scene.lower():
        pass
    else:
        raise NotImplementedError

    assert real_data_domain in ['in_place', 'out_of_place'], "real_data_domain={} is not supported!".format(
        real_data_domain)

    assert 1.0 >= real_data_chunk >= 0.0 and 1.0 >= sim_data_chunk >= 0.0

    assert real_data_chunk > 0.0 or sim_data_chunk > 0.0, "one of real_data_chunk or sim_data_chunk must be positive!"

    # semantics has fullsize immunity
    if task == 'semantics':
        _scene = scene
    else:
        _scene = scene + '-fullsize' if fullsize else scene

    # get all dataset folders to load
    dataset_to_load = []
    if sim_data_chunk > 0:
        if sim_data_chunk == 1:
            root_sim = "./datasets/" + _scene + "/train_sim"
        else:
            root_sim = "./datasets/" + _scene + "/train_sim_chunk_{:.2f}".format(sim_data_chunk)
        dataset_to_load.append(root_sim)
    if real_data_chunk > 0:
        if real_data_chunk == 1:
            if real_data_domain == 'in_place':
                root_real = "./datasets/" + _scene + "/train_drone_real"
                root_sim = "./datasets/" + _scene + "/train_drone_sim"
            elif real_data_domain == 'out_of_place':
                root_real = "./datasets/" + _scene + "/train_oop_drone_real"
                root_sim = "./datasets/" + _scene + "/train_oop_drone_sim"
            else:
                raise NotImplementedError
        else:
            if real_data_domain == 'in_place':
                root_real = "./datasets/" + _scene + "/train_drone_real_chunk_{:.2f}".format(real_data_chunk)
                root_sim = "./datasets/" + _scene + "/train_drone_sim_chunk_{:.2f}".format(real_data_chunk)
            elif real_data_domain == 'out_of_place':
                root_real = "./datasets/" + _scene + "/train_oop_drone_real_chunk_{:.2f}".format(real_data_chunk)
                root_sim = "./datasets/" + _scene + "/train_oop_drone_sim_chunk_{:.2f}".format(real_data_chunk)
            else:
                raise NotImplementedError
        dataset_to_load.append(root_real)
        if not real_only:
            dataset_to_load.append(root_sim)
    assert len(dataset_to_load)

    # original dataset to calculate mean
    trainset_vanilla = CamLocDataset("./datasets/" + _scene + "/train_sim", coord=True, depth=True, normal=True,
                                     semantics=False, augment=False, raw_image=False, mute=True, fullsize=fullsize)
    trainset_loader_vanilla = torch.utils.data.DataLoader(trainset_vanilla, shuffle=False, batch_size=1,
                                                          num_workers=mp.cpu_count() // 2, pin_memory=True,
                                                          collate_fn=trainset_vanilla.batch_resize)

    mean = get_label_mean(trainset_loader_vanilla, nodata_value, scene, task)
    flag_coord = task == 'coord'
    flag_depth = task == 'depth'
    flag_normal = task == 'normal'
    flag_semantics = task == 'semantics'

    trainset = CamLocDataset(dataset_to_load, coord=flag_coord, depth=flag_depth, normal=flag_normal,
                             semantics=flag_semantics,
                             augment=True, grayscale=grayscale, raw_image=False, fullsize=fullsize)
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=min(mp.cpu_count() // 2, 6),
                                                  pin_memory=True, collate_fn=trainset.batch_resize)

    if real_data_chunk == 0.0:
        logging.info("This training uses synthetic data only. {:d} iterations per epoch.".format(len(trainset)))
    else:
        if not real_only:
            logging.info(
                "This training uses pairwise sim-to-real data. {:d} iterations per epoch.".format(len(trainset)))
        else:
            logging.info("This training uses real data only. {:d} iterations per epoch.".format(len(trainset)))

    return trainset, trainset_loader, mean


def config_network(scene, task, tiny, grayscale, uncertainty, fullsize, mean,
                   learning_rate, no_lr_scheduling, auto_resume, epoch_plus, network_in, output_dir,
                   encoders_in=None, flag_reuse_coord_encoder=False, flag_unfreeze_coord_encoder=False):
    """
    Configure network and optimizer (task specific).
    """
    if 'urbanscape' in scene.lower() or 'naturescape' in scene.lower():
        if task == 'coord':
            num_task_channel = 3
        elif task == 'normal':
            num_task_channel = 2
        elif task == 'depth':
            num_task_channel = 1
        elif task == 'semantics':
            num_task_channel = 6
        else:
            raise NotImplementedError
        if uncertainty is None:
            num_pos_channel = 0
        elif uncertainty == 'MLE':
            num_pos_channel = 1
        else:
            raise NotImplementedError
        if task == 'semantics' and uncertainty is not None:
            raise NotImplementedError
        if task == 'semantics' and not fullsize:
            raise NotImplementedError

        if encoders_in is None:
            len_encoder = 0
        else:
            if flag_reuse_coord_encoder:
                len_encoder = len(encoders_in)
            else:
                assert not flag_unfreeze_coord_encoder
                len_encoder = len(encoders_in) - 1  # coord weight is only used for decoder initialization
        network = TransPoseNet(mean, tiny, grayscale, num_task_channel=num_task_channel,
                               num_pos_channel=num_pos_channel,
                               enc_add_res_block=2, dec_add_res_block=2, full_size_output=fullsize,
                               num_mlr=len_encoder, num_unfrozen_encoder=1 if flag_unfreeze_coord_encoder else 0)
        logging.info("{:d} network weights to load, flag_unfreeze_coord_encoder: {}".format(
            len_encoder, flag_unfreeze_coord_encoder))
    else:
        network = Network(mean, tiny)
    if network_in is not None:
        # load a single network, resume training or forceful loading
        network.load_state_dict(torch.load(network_in), strict=True)
        logging.info("Successfully loaded %s." % network_in)
        if auto_resume:
            model_path = os.path.join(output_dir, 'model_auto_resume.net')
        elif epoch_plus:
            model_path = os.path.join(output_dir, 'model_epoch_plus_resume.net')
        else:
            model_path = os.path.join(output_dir, 'model_resume.net')
        torch.save(network.state_dict(), model_path)
    if encoders_in is not None:
        flag_loading = True if network_in is None else False

        # load encoder networks, their parameters are frozen unless specified
        def _load_module(weight_to, weight_from, prefix='encoder.'):
            for key in weight_to.state_dict().keys():
                if key in weight_from.keys():
                    weight_to.state_dict()[key].copy_(weight_from[key])
                elif prefix + key in weight_from.keys():
                    weight_to.state_dict()[key].copy_(weight_from[prefix + key])
                else:
                    raise NotImplementedError

        enc_idx = 0
        for i, this_network_in in enumerate(encoders_in):
            if i == 0:
                # coord network
                assert 'coord' in os.path.abspath(this_network_in)  # make sure the fist weight is for coord task
                if flag_loading:
                    _load_module(network.decoder, torch.load(this_network_in),
                                 prefix='decoder.')  # coord for decoder init.
                if flag_reuse_coord_encoder:
                    # ues pretrained coord encoder
                    if flag_loading:
                        _load_module(network.mlr_encoder_ls[enc_idx], torch.load(this_network_in), prefix='encoder.')
                        logging.info(
                            "Successfully loaded coord pretrained model {:s} for decoder, reuse as encoder: {}, "
                            "unfreeze encoder parameter: {}".format(
                                this_network_in, flag_reuse_coord_encoder, flag_unfreeze_coord_encoder))
                    else:
                        logging.info("Skip loading coord network weight at {:s} because flag_loading is False".
                                     format(this_network_in))
                    if not flag_unfreeze_coord_encoder:
                        for param in network.mlr_encoder_ls[enc_idx].parameters():
                            param.requires_grad = False
                    enc_idx += 1
            else:
                # other networks, load and freeze the encoders
                if flag_loading:
                    _load_module(network.mlr_encoder_ls[enc_idx], torch.load(this_network_in), prefix='encoder.')
                    logging.info("Successfully loaded frozen pretrained model as an encoder %s." % this_network_in)
                else:
                    logging.info("Skip loading encoder weight at {:s} because flag_loading is False".
                                 format(this_network_in))
                for param in network.mlr_encoder_ls[enc_idx].parameters():
                    param.requires_grad = False
                enc_idx += 1
        model_path = os.path.join(output_dir, 'model.net')
        torch.save(network.state_dict(), model_path)
        logging.info("Saving the initialized MLR model weight to {:s}".format(model_path))

    if network_in is None and encoders_in is None:
        model_path = os.path.join(output_dir, 'model.net')

    # recount trainable parameters
    if encoders_in is not None:
        ttl_num_param = 0
        param_info = 'Recounting #trainable parameters: '
        for name, struct in zip(['Vanilla encoder', 'MLR encoder', 'Decoder'],
                                [network.encoder_ls, network.mlr_ls, network.decoder_ls]):
            num_param = sum([param.numel() for layer in struct for param in layer.parameters() if param.requires_grad])
            ttl_num_param += num_param
            param_info += '{:s}: {:,d}, '.format(name, num_param)
        param_info += 'Total: {:,d}.'.format(ttl_num_param)
        logging.info(param_info)

    network = network.cuda()
    network.train()

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    if no_lr_scheduling:
        # scheduler would be actually turned OFF
        scheduler = MultiStepLR(optimizer, [999999], gamma=1.0)
    else:
        scheduler = MultiStepLR(optimizer, [50, 100], gamma=0.5)

    return network, optimizer, model_path, scheduler


def xyz2ae(xyz: torch.Tensor) -> torch.Tensor:
    """
    Turn normalized direction vector into azimuth and elevation.
    @param xyz  [B, 3, *] tensor of normalized direction vector.
    @return:    [B, 2, *] tensor of azimuth and elevation in radian.
    """

    # azimuth = arctan2(y, x), range [-pi, pi]
    azimuth = torch.atan2(xyz[:, 1], xyz[:, 0])  # [B, *]

    # elevation = arctan2(z, sqrt(x**2 + y**2)), range [-pi, pi]
    elevation = torch.atan2(xyz[:, 2], torch.norm(xyz[:, 0:2], dim=1, p=2))  # [B, *]

    return torch.stack([azimuth, elevation], dim=1)  # [B, 2, *]


def ae2xyz(ae: torch.Tensor) -> torch.Tensor:
    """
    Turn azimuth and elevation into normalized direction vector.
    @param ae   [B, 2, *] tensor of azimuth and elevation in radian.
    @return:    [B, 3, *] tensor of normalized direction vector.
    """
    XY_norm = torch.cos(ae[:, 1])  # [B, *]
    X = torch.cos(ae[:, 0]) * XY_norm  # [B, *]
    Y = torch.sin(ae[:, 0]) * XY_norm  # [B, *]
    Z = torch.sin(ae[:, 1])  # [B, *]
    XYZ = torch.stack([X, Y, Z], dim=1)  # [B, 3, *]
    return nn.functional.normalize(XYZ, p=2, dim=1)


def logits_to_radian(activation: torch.Tensor) -> torch.Tensor:
    """
    Convert the arbitrary activation into [-pi, pi] radian angle.
    @param activation: tensor of any size
    @return:
    """
    # radian = torch.tanh(activation) * np.pi
    radian = torch.sigmoid(activation).clamp(min=1.e-7, max=1 - 1.e-7)  # range [0, 1]
    radian = (radian * 2 - 1.0) * np.pi  # range [-pi, pi]
    return radian
