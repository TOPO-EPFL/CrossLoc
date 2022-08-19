import os
import shutil
import logging
import sys
import git
import re
import math
import random
import logging
from datetime import datetime

import cv2
import torch
import torch.multiprocessing as mp
import numpy as np
import pdb

from networks.gadgets import logits_to_radian, xyz2ae, ae2xyz


def _safe_printout(words):
    """
    Safely printout the string whenever the logging handler is enabled.
    @param words:
    @return:
    """
    if logging.getLogger().hasHandlers():
        logging.info(words)
    else:
        print(words)


def read_training_log(log_path, iter_per_epoch):
    """
    Read training log and analyze the last 100 lines to retrieve training status.
    """

    # read training status
    with open(log_path, 'r') as f:
        lines_100 = f.readlines()[-100:]
        lines_100 = ''.join(lines_100)

    pattern = r"Iteration:\s+(?P<iter>\d+), Epoch:\s+(?P<epoch>\d+)"

    iterations_ls = [int(item[0]) for item in re.findall(pattern, lines_100)]
    epochs_ls = [int(item[1]) for item in re.findall(pattern, lines_100)]

    last_iteration = np.max(iterations_ls)
    last_epoch = np.max(epochs_ls)

    # sanity check for the read statistics
    assert abs(last_iteration // iter_per_epoch - last_epoch) <= 1, "Last iteration {:d} does not match last epoch {:d} with iteration per epoch being {:d}.".format(last_iteration, last_epoch, iter_per_epoch)
    return last_iteration, last_epoch


def load_geo_dist(geo_dist_path, retain_geo_dist=False):
    """
    Load pre-computed geometric distance table.
    @param geo_dist_path:    Path to the geometric distance dict.
    @param retain_geo_dist:  Flag to retain the original geo-distance tensor.
    """
    assert os.path.exists(geo_dist_path)
    _safe_printout('Load geometric distance from {:s}...'.format(geo_dist_path))
    geo_dist_dict = torch.load(geo_dist_path)
    if retain_geo_dist:
        geo_dist = geo_dist_dict['geo_dist'].cpu()  # raw distance table, debug purpose
    else:
        geo_dist = None
    sim_data = geo_dist_dict['sim_data'].cpu()
    dict_name_to_idx = geo_dist_dict['dict_name_to_idx']
    dict_idx_to_name = geo_dist_dict['dict_idx_to_name']
    hyper_params = geo_dist_dict['hyper_params']
    dict_supercon = geo_dist_dict['dict_supercon']
    feasible_anchor = geo_dist_dict['feasible_anchor']

    # sanity check for index-based synthetic data flag
    for i in range(len(sim_data)):
        if i < sim_data.sum().item():
            assert sim_data[i]
        else:
            assert ~sim_data[i]

    _safe_printout('%d / %d samples are feasible anchors after pre-screening.' % (len(feasible_anchor), len(sim_data)))

    return geo_dist, sim_data, dict_name_to_idx, dict_idx_to_name, hyper_params, dict_supercon, feasible_anchor


def get_supercon_dataloader(trainset_supercon, shuffle=True):
    """
    Wrapper to reset the supercon dataloder.
    """
    sampler_supercon = trainset_supercon.get_supercon_sampler(shuffle=shuffle)
    loader_supercon = torch.utils.data.DataLoader(trainset_supercon, batch_sampler=sampler_supercon,
                                                  num_workers=mp.cpu_count() // 2,
                                                  pin_memory=True, collate_fn=trainset_supercon.batch_resize)
    return loader_supercon


def get_unique_file_name(file_path):
    """
    Get unique file name for unique mapping.
    The generated filename includes basename and section, e.g., EPFL-LHS_00000_LHS.png@train_sim_aug
    """
    file_section = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    unique_file_nm = os.path.basename(file_path) + '@' + file_section
    return unique_file_nm


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


def get_cam_mat(width, height, focal_length):
    """
    Get intrinsic camera matrix (tensor)
    """
    cam_mat = torch.eye(3)
    cam_mat[0, 0] = focal_length
    cam_mat[1, 1] = focal_length
    cam_mat[0, 2] = width / 2
    cam_mat[1, 2] = height / 2
    cam_mat = cam_mat.cuda()
    return cam_mat


def get_nodata_value(scene_name):
    """
    Get nodata value based on dataset scene name.
    """
    if '7scenes' in scene_name or '12scenes' in scene_name or 'Cambridge' in scene_name:
        nodata_value = 0
    elif 'epfl' in scene_name.lower() or 'comballaz' in scene_name.lower():
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


def coords_world_to_cam(scene_coords, gt_coords, gt_poses):
    """
    Transform the scene coordinates to camera coordinates.
    @param scene_coords           [B, 3, N] Predicted scene coords tensor.
    @param gt_coords              [B, 3, N] Ground-truth scene coords tensor.
    @param gt_poses               [B, 4, 4] cam-to-world matrix.
    @return camera_coords         [B, 3, N] camera coords tensor corresponding to scene_coords.
    @return target_camera_coords  [B, 3, N] camera coords tensor corresponding to gt_coords.
    """
    gt_pose_inv = gt_poses.inverse()[:, 0:3, :]  # [B, 3, 4], world to camera matrix
    ones = torch.ones((scene_coords.size(0), 1, scene_coords.size(2))).cuda()

    scene_coords_ = torch.cat([scene_coords, ones], dim=1)  # [B, 4, N]
    gt_coords_ = torch.cat([gt_coords, ones], dim=1)  # [B, 4, N]

    camera_coords = torch.bmm(gt_pose_inv, scene_coords_)  # [B, 3, N] = [B, 3, 4] * [B, 4, N]
    target_camera_coords = torch.bmm(gt_pose_inv, gt_coords_)  # [B, 3, N] = [B, 3, 4] * [B, 4, N]

    return camera_coords, target_camera_coords


def get_repro_err(camera_coords, cam_mat, pixel_grid_crop, min_depth):
    """
    Get reprojection error for each pixel.
    @param camera_coords        [B, 3, N] tensor for camera coordinates.
    @param cam_mat              [3, 3] tensor for intrinsic camera matrix.
    @param pixel_grid_crop      [2, N] tensor for pixel grid.
    @param min_depth            Scalar for minimum reprojected depth.
    @return reprojection_error  [B, N] tensor for reprojection error in pixel.
    """
    batch_size = camera_coords.size(0)
    reprojection_error = torch.bmm(cam_mat.expand(batch_size, -1, -1), camera_coords)  # [B, 3, H_ds*W_ds]
    reprojection_error[:, 2].clamp_(min=min_depth)  # avoid division by zero
    reprojection_error = reprojection_error[:, 0:2] / reprojection_error[:, 2:]  # [B, 2, H_ds*W_ds]

    reprojection_error = reprojection_error - pixel_grid_crop[None, :, :]
    reprojection_error = reprojection_error.norm(p=2, dim=1).clamp(min=1.e-7)  # [B, H_ds*W*ds]
    return reprojection_error


def check_constraints(camera_coords, reproj_error, cam_coords_reg_error, mask_gt_coords_nodata,
                      min_depth, max_reproj_error, max_coords_reg_error):
    """
    Check constraints on network prediction.
    @param  camera_coords          [B, 3, N] tensor for camera coordinates.
    @param  reproj_error           [B, N] tensor for reprojection errors.
    @param  cam_coords_reg_error   [B, N] tensor for scene coordinate regression raw errors including invalid points.
    @param  mask_gt_coords_nodata  [B, N] tensor indicating points w/o valid scene coords labels.
    @param  min_depth              Scalar, threshold of minimum depth before camera panel in meter.
    @param  max_reproj_error       Scalar, threshold of maximum reprojection error in pixel.
    @param  max_coords_reg_error   Scalar, threshold of maximum scene coords regression error in meter.
    @return valid_sc               [B, N] Pixels w/ valid scene coords prediction, goes for reprojection error.
    """
    # check predicted scene coordinate for various constraints
    invalid_min_depth = camera_coords[:, 2] < min_depth  # [B, N], behind or too close to camera plane
    invalid_repro = reproj_error > max_reproj_error      # [B, N], very large reprojection errors

    # check for additional constraints regarding ground truth scene coordinates
    invalid_gt_distance = cam_coords_reg_error > max_coords_reg_error  # [B, N] too far from ground truth scene coordinates
    invalid_gt_distance[mask_gt_coords_nodata] = 0  # [B, N], filter out unknown ground truth scene coordinates

    # combine all constraints
    valid_sc = (invalid_min_depth + invalid_repro + invalid_gt_distance) == 0  # [B, N]

    return valid_sc


def get_pose_err(pose_gt, pose_est):
    """
    Compute translation and rotation error between two 4x4 transformation matrices.
    """
    if isinstance(pose_gt, np.ndarray):
        assert isinstance(pose_est, np.ndarray)
        transl_err = np.linalg.norm(pose_gt[0:3, 3] - pose_est[0:3, 3])
        rot_err = pose_est[0:3, 0:3].T.dot(pose_gt[0:3, 0:3])
        rot_err = cv2.Rodrigues(rot_err)[0]
        rot_err = np.reshape(rot_err, (1, 3))
        rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.
        rot_err = rot_err[0]
    elif isinstance(pose_gt, torch.Tensor):
        assert isinstance(pose_est, torch.Tensor)
        transl_err = torch.norm(pose_gt[0:3, 3] - pose_est[0:3, 3])
        rot_err = torch.mm(pose_est[0:3, 0:3].t(), pose_gt[0:3, 0:3])

        # 1) don't use kn.rotation_matrix_to_angle_axis as it is wrong!
        # 2) pytorch3d.transforms seems unstable and is deprecated (may work in future pytorch3d ver.)
        # rot_err = transforms.matrix_to_quaternion(rot_err)
        # rot_err = transforms.quaternion_to_axis_angle(rot_err).view(-1)
        # 3) classical & stable matrix-angle conversion: acos((tr(R_12) - 1) / 2)
        safe_acos_input = torch.clamp((torch.trace(rot_err) - 1) / 2, min=-1 + 1.e-7, max=1 - 1.e-7)
        rot_err = torch.acos(safe_acos_input)
        rot_err = torch.norm(rot_err) / np.pi * 180.
    else:
        raise NotImplementedError
    return transl_err, rot_err


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
        if 'comballaz' in scene:
            mean = torch.tensor([-382.85, 430.71, 449.71]).float()
        elif 'EPFL' in scene:
            mean = torch.tensor([-31.31, 184.46, 92.98]).float()
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
        if 'comballaz' in scene:
            mean = torch.tensor([311.39]).float()
        elif 'EPFL' in scene:
            mean = torch.tensor([136.91]).float()
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

        if 'comballaz' in scene:
            mean = (torch.tensor([-0.8816, -0.8308]) / np.pi + 1.0) / 2.0  # angles to raw sigmoid output
            mean = _inverse_sigmoid(mean).float()
        elif 'EPFL' in scene:
            mean = (torch.tensor([-1.2142, -0.9927]) / np.pi + 1.0) / 2.0  # angles to raw sigmoid output
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
    else:
        raise NotImplementedError

    return mean


def scene_coords_regression_loss(mindepth, softclamp, hardclamp, inittolerance, uncertainty,
                                 pixel_grid, nodata_value, cam_mat,
                                 scene_coords, uncertainty_map, gt_poses, gt_coords, reduction='mean'):
    """
    Calculate scene coordinate regression loss, based on DSAC* and KF-Net implementation.
    Code: https://github.com/vislearn/dsacstar
    Paper: https://arxiv.org/abs/2002.12324

    @param mindepth             Scalar hyper-parameter for minimum reprojected depth.
    @param softclamp            Scalar hyper-parameter for loss clamp.
    @param hardclamp            Scalar hyper-parameter for loss clamp.
    @param inittolerance        Scalar hyper-parameter for coordinate regression loss.
    @param uncertainty          Flag for uncertainty loss.
    @param pixel_grid           [2, M, N] Pixel positions tensor.
    @param nodata_value         Scalar to indicate NODATA element of ground truth scene coordinates.
    @param cam_mat              [3, 3] tensor for intrinsic camera matrix.
    @param scene_coords         [B, 3, H_ds, W_ds] Predicted scene coords tensor.
    @param uncertainty_map      [B, 1, H_ds, W_ds] Uncertainty map tensor.
    @param gt_poses             [B, 4, 4] Camera to world matrix
    @param gt_coords            [B, 3, H_ds, W_ds] ---> [B, 3, 60, 80] by default w/o augmentation
    @param reduction            Method to post-process the mini-batch loss, 'mean' for mean and None for not aggregating
    @return loss                Regression loss value.
    @return num_valid_sc_rate   Rate of valid scene coordinates.
    """

    """RGB mode, optimize a variant of the reprojection error"""
    # crop ground truth pixel positions to prediction size
    pixel_grid_crop = pixel_grid[:, 0:scene_coords.size(2), 0:scene_coords.size(3)].clone().view(2, -1)

    scene_coords = scene_coords.view(scene_coords.size(0), 3, -1)  # [B, 3, H_ds*W_ds]
    gt_coords = gt_coords.view(gt_coords.size(0), 3, -1)  # [B, 3, H_ds*W_ds]

    camera_coords, target_camera_coords = coords_world_to_cam(scene_coords, gt_coords, gt_poses)  # [B, 3, H_ds*W_ds]*2
    camera_coords_reg_error = torch.norm(camera_coords - target_camera_coords, dim=1, p=2)  # [B, H_ds*W_ds]

    reprojection_error = get_repro_err(camera_coords, cam_mat, pixel_grid_crop, mindepth)  # [B, H_ds*W_ds]

    # check for invalid/unknown ground truth scene coordinates
    mask_gt_coords_valdata = pick_valid_points(gt_coords[:, :3, :], nodata_value, boolean=True)  # [B, H_ds*W_ds]
    mask_gt_coords_nodata = torch.logical_not(mask_gt_coords_valdata)  # [B, H_ds*W_ds]

    valid_scene_coordinates = check_constraints(
        camera_coords, reprojection_error, camera_coords_reg_error, mask_gt_coords_nodata,
        min_depth=mindepth, max_reproj_error=hardclamp,
        max_coords_reg_error=inittolerance)  # [B, H_ds*W_ds], warning: it is not coupled with mask_gt_coords_valdata!
    invalid_scene_coordinates = torch.logical_not(valid_scene_coordinates)  # [B, H_ds*W_ds]
    num_valid_sc = valid_scene_coordinates.sum(dim=1).cpu().numpy()  # [B]
    num_pixels_batch = valid_scene_coordinates.numel()  # number of all pixels in the batch
    num_pixels_instance = valid_scene_coordinates[0].numel()  # number of pixels in one data point

    # assemble loss
    loss = 0

    """Reprojection error for all valid scene coordinates"""
    if num_valid_sc.sum() > 0:
        # calculate soft clamped l1 loss of reprojection error
        reprojection_error = reprojection_error * valid_scene_coordinates  # [B, H_ds*W_ds]
        loss_l1 = torch.sum(reprojection_error * (reprojection_error <= softclamp), dim=1).clamp(min=1.e-7)  # [B]
        loss_sqrt = reprojection_error * (reprojection_error > softclamp)  # [B, H_ds*W_ds]
        loss_sqrt = torch.sum(torch.sqrt(softclamp * loss_sqrt + 1.e-7), dim=1).clamp(min=1.e-7)  # [B]
        loss += loss_l1 + loss_sqrt  # [B]

    """3D distance loss for all invalid scene coordinates where the ground truth is known"""
    if uncertainty:
        uncertainty_map = uncertainty_map.view(uncertainty_map.size(0), -1).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        gt_coord_dist = torch.norm(scene_coords - gt_coords, dim=1, p=2).square().clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_unc = 3.0 * torch.log(uncertainty_map) + gt_coord_dist / (
                2.0 * uncertainty_map.square().clamp(min=1.e-7))  # [B, H_ds*W_ds]
        loss += torch.sum(loss_unc * mask_gt_coords_valdata, dim=1)  # [B], applied to all pixels w/ valid labels
    else:
        invalid_scene_coordinates[mask_gt_coords_nodata] = 0  # filter out pixels w/o valid labels
        loss += torch.sum(camera_coords_reg_error * invalid_scene_coordinates,
                          dim=1)  # [B], applied to invalid pixels w/ valid labels

    valid_pred_rate = num_valid_sc.sum() / num_pixels_batch  # scalar

    if reduction is None:
        loss /= num_pixels_instance  # [B], each item is the mean over all pixels within one instance
    elif reduction == 'mean':
        loss = loss.sum()  # scalar, mean over each pixels within the batch
        loss /= num_pixels_batch
    else:
        raise NotImplementedError
    return loss, valid_pred_rate


def depth_regression_loss(mindepth, softclamp, hardclamp, uncertainty,
                          nodata_value, depth_map, uncertainty_map, gt_depths, reduction='mean'):
    """
    Calculate depth regression loss.

    @param mindepth             Scalar hyper-parameter for minimum reprojected depth.
    @param softclamp            Scalar hyper-parameter for loss clamp.
    @param hardclamp            Scalar hyper-parameter for loss clamp.
    @param uncertainty          Flag for uncertainty loss.
    @param nodata_value         Scalar to indicate NODATA element of ground truth scene coordinates.
    @param depth_map            [B, 1, H_ds, W_ds] Predicted depth tensor.
    @param uncertainty_map      [B, 1, H_ds, W_ds] Uncertainty map tensor.
    @param gt_depths            [B, 1, H_ds, W_ds] ---> [B, 1, 60, 80] by default w/o augmentation
    @param reduction            Method to post-process the mini-batch loss, 'mean' for mean and None for not aggregating
    @return loss                Regression loss value.
    @return num_valid_sc_rate   Rate of valid scene coordinates.
    """

    depth_map = depth_map.view(depth_map.size(0), -1)  # [B, H_ds*W_ds]
    gt_depths = gt_depths.view(depth_map.size(0), -1)  # [B, H_ds*W_ds]

    depth_reg_error = torch.abs(depth_map - gt_depths)  # [B, H_ds*W_ds]

    # check for invalid/unknown ground truth scene coordinates
    mask_gt_coords_valdata = pick_valid_points(gt_depths.unsqueeze(1), nodata_value, boolean=True)  # [B, H_ds*W_ds]
    mask_gt_coords_nodata = torch.logical_not(mask_gt_coords_valdata)  # [B, H_ds*W_ds]

    """check predicted depth for various constraints"""
    invalid_min_depth = depth_map < mindepth  # [B, H_ds*W_ds], behind or too close to camera plane

    # check for additional constraints regarding ground truth scene coordinates
    invalid_gt_distance = depth_reg_error > hardclamp  # [B, H_ds*W_ds] too far from ground truth scene coordinates

    # combine all constraints
    valid_depth = (invalid_min_depth + invalid_gt_distance + mask_gt_coords_nodata) == 0  # [B, N]

    invalid_depth = torch.logical_not(valid_depth)  # [B, H_ds*W_ds]
    num_valid_depth = valid_depth.sum(dim=1).cpu().numpy()  # [B]
    num_pixels_batch = valid_depth.numel()
    num_pixels_instance = valid_depth[0].numel()

    # assemble loss
    loss = 0

    """Clamped loss for all valid depth predictions"""
    if num_valid_depth.sum() > 0:
        # calculate soft clamped l1 loss for depth
        depth_reg_error_valid = depth_reg_error * valid_depth  # [B, H_ds*W_ds]
        loss_l1 = torch.sum(depth_reg_error_valid * (depth_reg_error_valid <= softclamp), dim=1).clamp(min=1.e-7)  # [B]
        loss_sqrt = depth_reg_error_valid * (depth_reg_error_valid > softclamp)  # [B, H_ds*W_ds]
        loss_sqrt = torch.sum(torch.sqrt(softclamp * loss_sqrt + 1.e-7), dim=1).clamp(min=1.e-7)  # [B]
        loss += loss_l1 + loss_sqrt  # [B]

    """3D distance loss for all invalid pixels where the ground truth is known"""
    if uncertainty:
        uncertainty_map = uncertainty_map.view(uncertainty_map.size(0), -1).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        depth_reg_error = depth_reg_error.square().clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_unc = 3.0 * torch.log(uncertainty_map) + depth_reg_error / (
                2.0 * uncertainty_map.square().clamp(min=1.e-7))  # [B, H_ds*W_ds]
        loss += torch.sum(loss_unc * mask_gt_coords_valdata, dim=1)  # [B], applied to all pixels w/ valid labels
    else:
        invalid_depth[mask_gt_coords_nodata] = 0  # filter out pixels w/o valid labels
        loss += torch.sum(depth_reg_error * invalid_depth, dim=1)  # [B], applied to invalid pixels w/ valid labels

    valid_pred_rate = num_valid_depth.sum() / num_pixels_batch  # scalar

    if reduction is None:
        loss /= num_pixels_instance  # [B], each item is the mean over all pixels within one instance
    elif reduction == 'mean':
        loss = loss.sum()  # scalar, mean over each pixels within the batch
        loss /= num_pixels_batch
    else:
        raise NotImplementedError
    return loss, valid_pred_rate


def normal_regression_loss(softclamp, hardclamp, uncertainty,
                           nodata_value, normal_logits, uncertainty_map, gt_normals, reduction='mean'):
    """
    Calculate surface normal regression loss. The loss is extracted in the azimuth-elevation mode for stability.

    Reference: Deep Surface Normal Estimation on the 2-Sphere with Confidence Guided Semantic Attention, 2020 ECCV
    Link: https://doi.org/10.1007/978-3-030-58586-0_43

    @param softclamp            Scalar hyper-parameter for loss clamp.
    @param hardclamp            Scalar hyper-parameter for loss clamp.
    @param uncertainty          Flag for uncertainty loss.
    @param nodata_value         Scalar to indicate NODATA element of ground truth scene coordinates.
    @param normal_logits        [B, 2, H_ds, W_ds] Predicted surface normal logits tensor.
    @param uncertainty_map      [B, 1, H_ds, W_ds] Uncertainty map tensor.
    @param gt_normals           [B, 3, H_ds, W_ds] ---> [B, 3, 60, 80] by default w/o augmentation
    @param reduction            Method to post-process the mini-batch loss, 'mean' for mean and None for not aggregating
    @return loss                Regression loss value.
    @return num_valid_sc_rate   Rate of valid scene coordinates.
    """

    """Initialization"""
    batch_size = normal_logits.size(0)
    normal_logits = normal_logits.view(batch_size, 2, -1)  # [B, 2, H_ds*W_ds]
    gt_normals = gt_normals.view(batch_size, 3, -1)  # [B, 3, H_ds*W_ds]

    """Compute spherical normal regression error"""
    pred_normals_ae = logits_to_radian(normal_logits)  # [B, 2, H_ds*W_ds], range [-pi, pi]
    gt_normals_ae = xyz2ae(gt_normals)  # [B, 2, H_ds*W_ds], range [-pi, pi]

    def _azimuth_circle_loss(azimuth_pred, azimuth_gt):
        """Circle loss for azimuth prediction"""
        l1_error = torch.abs(azimuth_gt - azimuth_pred)
        return 2.0 * torch.min(l1_error, 2.0 * np.pi - l1_error).abs()

    azimuth_loss = _azimuth_circle_loss(pred_normals_ae[:, 0], gt_normals_ae[:, 0])  # [B, H_ds*W_ds]
    elevation_loss = torch.abs(pred_normals_ae[:, 1] - gt_normals_ae[:, 1])  # [B, H_ds*W_ds]
    normal_reg_error = (azimuth_loss + elevation_loss).clamp(min=1.e-7)  # [B, H_ds*W_ds]

    """DEBUG"""
    if torch.isnan(normal_reg_error).sum():
        _safe_printout("normal_logits has {:d} / {:d} nan values!".format(torch.isnan(normal_logits).sum(),
                                                                          normal_logits.numel()))
        _safe_printout("normal_reg_error has {:d} / {:d} nan values!".format(torch.isnan(normal_reg_error).sum(),
                                                                             normal_reg_error.numel()))
        _safe_printout("azimuth_loss has {:d} / {:d} nan values!".format(torch.isnan(azimuth_loss).sum(),
                                                                         azimuth_loss.numel()))
        _safe_printout("elevation_loss has {:d} / {:d} nan values!".format(torch.isnan(elevation_loss).sum(),
                                                                           elevation_loss.numel()))
        _safe_printout("pred_normals_ae has {:d} / {:d} nan values!".format(torch.isnan(pred_normals_ae).sum(),
                                                                            pred_normals_ae.numel()))
        _safe_printout("gt_normals_ae has {:d} / {:d} nan values!".format(torch.isnan(gt_normals_ae).sum(),
                                                                          gt_normals_ae.numel()))
    """"""

    # check for invalid/unknown ground truth scene coordinates
    mask_gt_coords_valdata = pick_valid_points(gt_normals, nodata_value, boolean=True)  # [B, H_ds*W_ds]
    mask_gt_coords_nodata = torch.logical_not(mask_gt_coords_valdata)  # [B, H_ds*W_ds]

    """Check predicted depth for various constraints"""
    # check for additional constraints regarding ground truth scene coordinates
    pred_normals_xyz = ae2xyz(pred_normals_ae.clone().detach())  # [B, 3, H_ds*W_ds]
    gt_normals_xyz = gt_normals  # [B, 3, H_ds*W_ds]
    normal_angle_error = torch.nn.functional.cosine_similarity(pred_normals_xyz, gt_normals_xyz, dim=1)  # [B, H_ds*W_ds], range [-1, 1]
    normal_angle_error = torch.acos(normal_angle_error.clamp(min=-1 + 1.e-7, max=1 - 1.e-7))  # [B, H_ds*W_ds], range [0, pi]
    normal_angle_error = normal_angle_error / np.pi * 180.0  # range [0, 180], unit in degrees

    invalid_gt_angle = normal_angle_error > hardclamp  # [B, H_ds*W_ds] too far from ground truth normals

    """DEBUG"""
    if torch.isnan(normal_angle_error).sum():
        _safe_printout("pred_normals_xyz has {:d} / {:d} nan values!".format(torch.isnan(pred_normals_xyz).sum(),
                                                                             pred_normals_xyz.numel()))
        _safe_printout("normal_angle_error has {:d} / {:d} nan values!".format(torch.isnan(normal_angle_error).sum(),
                                                                               normal_angle_error.numel()))
        cos_sim = torch.nn.functional.cosine_similarity(pred_normals_xyz, gt_normals_xyz, dim=1)
        _safe_printout("cos_sim has {:d} / {:d} nan values!".format(torch.isnan(cos_sim).sum(), cos_sim.numel()))

        arc_cos = torch.acos(cos_sim.clamp(min=-1 + 1.e-7, max=1 - 1.e-7))
        _safe_printout("arc_cos has {:d} / {:d} nan values!".format(torch.isnan(arc_cos).sum(), arc_cos.numel()))
    """"""

    # combine all constraints
    valid_normal = (invalid_gt_angle + mask_gt_coords_nodata) == 0  # [B, H_ds*W_ds]
    invalid_normal = torch.logical_not(valid_normal)  # [B, N]
    num_valid_normal = valid_normal.sum(dim=1).cpu().numpy()  # [B]
    num_pixels_batch = valid_normal.numel()  # B*N
    num_pixels_instance = valid_normal[0].numel()

    # assemble loss
    loss = 0

    """Clamped loss for all valid depth predictions"""
    if num_valid_normal.sum() > 0:
        # calculate soft clamped l1 loss for depth
        normal_sphere_loss_valid = normal_reg_error * valid_normal  # [B, H_ds*W_ds]
        loss_l1 = torch.sum(normal_sphere_loss_valid * (normal_angle_error <= softclamp), dim=1).clamp(min=1.e-7)  # [B]
        loss_sqrt = normal_sphere_loss_valid * (normal_angle_error > softclamp)  # [B, H_ds*W_ds]
        loss_sqrt = torch.sum(torch.sqrt(softclamp * loss_sqrt + 1.e-7), dim=1).clamp(min=1.e-7)  # [B]
        loss += loss_l1 + loss_sqrt  # [B]

    """uncertainty loss for all invalid pixels where the ground truth is known"""
    if uncertainty:
        uncertainty_map = uncertainty_map.view(uncertainty_map.size(0), -1).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        normal_reg_error = normal_reg_error.square().clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_unc = 3.0 * torch.log(uncertainty_map) + normal_reg_error / (
                2.0 * uncertainty_map.square().clamp(min=1.e-7))  # [B, H_ds*W_ds]
        loss += torch.sum(loss_unc * mask_gt_coords_valdata, dim=1)  # [B], applied to all pixels w/ valid labels
    else:
        invalid_normal[mask_gt_coords_nodata] = 0  # filter out pixels w/o valid labels
        loss += torch.sum(normal_reg_error * invalid_normal, dim=1)  # [B], applied to invalid pixels w/ valid labels

    valid_pred_rate = num_valid_normal.sum() / num_pixels_batch  # scalar

    if reduction is None:
        loss /= num_pixels_instance  # [B], each item is the mean over all pixels within one instance
    elif reduction == 'mean':
        loss = loss.sum()  # scalar, mean over each pixels within the batch
        loss /= num_pixels_batch
    else:
        raise NotImplementedError
    return loss, valid_pred_rate
