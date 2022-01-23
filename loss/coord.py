import torch
import pdb
from utils.learning import pick_valid_points
from utils.io import safe_printout


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
    invalid_gt_distance = cam_coords_reg_error > max_coords_reg_error  # [B, N] too far from ground truth
    invalid_gt_distance[mask_gt_coords_nodata] = 0  # [B, N], filter out unknown ground truth

    # combine all constraints
    valid_sc = (invalid_min_depth + invalid_repro + invalid_gt_distance) == 0  # [B, N]

    return valid_sc


def scene_coords_regression_loss(min_depth, soft_clamp, hard_clamp, init_tolerance, uncertainty,
                                 pixel_grid, nodata_value, cam_mat,
                                 scene_coords, uncertainty_map, gt_poses, gt_coords, reduction='mean'):
    """
    Calculate scene coordinate regression loss, based on DSAC* and KF-Net implementation.
    Code: https://github.com/vislearn/dsacstar
    Paper: https://arxiv.org/abs/2002.12324

    @param min_depth            Scalar hyper-parameter for minimum reprojected depth.
    @param soft_clamp           Scalar hyper-parameter for loss clamp.
    @param hard_clamp           Scalar hyper-parameter for loss clamp.
    @param init_tolerance       Scalar hyper-parameter for coordinate regression loss.
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

    reprojection_error = get_repro_err(camera_coords, cam_mat, pixel_grid_crop, min_depth)  # [B, H_ds*W_ds]

    # check for invalid/unknown ground truth scene coordinates
    mask_gt_coords_valdata = pick_valid_points(gt_coords[:, :3, :], nodata_value, boolean=True)  # [B, H_ds*W_ds]
    mask_gt_coords_nodata = torch.logical_not(mask_gt_coords_valdata)  # [B, H_ds*W_ds]

    valid_scene_coordinates = check_constraints(
        camera_coords, reprojection_error, camera_coords_reg_error, mask_gt_coords_nodata,
        min_depth=min_depth, max_reproj_error=hard_clamp,
        max_coords_reg_error=init_tolerance)  # [B, H_ds*W_ds], warning: it is not coupled with mask_gt_coords_valdata!
    num_valid_sc = valid_scene_coordinates.sum(dim=1).cpu().numpy()  # [B]
    num_pixels_batch = valid_scene_coordinates.numel()  # number of all pixels in the batch
    num_pixels_instance = valid_scene_coordinates[0].numel()  # number of pixels in one data point

    # assemble loss
    loss = 0

    """Reprojection error for all valid scene coordinates"""
    loss_reproj = 0
    if num_valid_sc.sum() > 0:
        # calculate soft clamped l1 loss of reprojection error
        # it's only applied to the **valid** scene coordinates
        reprojection_error = reprojection_error * valid_scene_coordinates  # [B, H_ds*W_ds], masked
        loss_l1 = (reprojection_error * (reprojection_error <= soft_clamp)).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_sqrt = (reprojection_error * (reprojection_error > soft_clamp)).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_sqrt = torch.sqrt(soft_clamp * loss_sqrt + 1.e-7).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_reproj = loss_l1 + loss_sqrt  # [B, H_ds*W_ds]

    """3D distance loss for pixels whose ground truth is known"""
    # assemble loss
    if uncertainty is None:
        # original DSAC* implementation, not used
        # invalid_scene_coordinates = torch.logical_not(valid_scene_coordinates)  # [B, H_ds*W_ds]
        # invalid_scene_coordinates[mask_gt_coords_nodata] = 0  # filter out pixels w/o valid labels
        # loss += torch.sum(camera_coords_reg_error * invalid_scene_coordinates,
        #                   dim=1)  # [B], applied to invalid pixels w/ valid labels

        # L2 distance loss on both coordinate regression and reprojection error
        # it's applied to all pixels w/ valid labels
        loss += torch.sum(camera_coords_reg_error * mask_gt_coords_valdata + loss_reproj, dim=1)  # [B]
    elif uncertainty == 'MLE':
        uncertainty_map = uncertainty_map.view(uncertainty_map.size(0), -1).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        coord_error_square = camera_coords_reg_error.square().clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_unc = 3.0 * torch.log(uncertainty_map) + coord_error_square / (
                2.0 * uncertainty_map.square().clamp(min=1.e-7))  # [B, H_ds*W_ds]
        loss += torch.sum(loss_unc * mask_gt_coords_valdata + loss_reproj, dim=1)  # [B]

        # diagnosis
        safe_printout(
            'Regression error: coord:  %.2f, reprojection:  %.2f' % (
                torch.sum(camera_coords_reg_error * mask_gt_coords_valdata).item()
                / max(1, mask_gt_coords_valdata.sum().item()),
                torch.sum(reprojection_error * valid_scene_coordinates).item()
                / max(1, valid_scene_coordinates.sum().item())))
    else:
        raise NotImplementedError

    valid_pred_rate = num_valid_sc.sum() / num_pixels_batch  # scalar

    if reduction is None:
        loss /= num_pixels_instance  # [B], each item is the mean over all pixels within one instance
    elif reduction == 'mean':
        loss = loss.sum()  # scalar, mean over each pixels within the batch
        loss /= num_pixels_batch
    else:
        raise NotImplementedError
    return loss, valid_pred_rate

