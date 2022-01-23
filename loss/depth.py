import torch
from utils.learning import pick_valid_points
from utils.io import safe_printout
import pdb


def depth_regression_loss(min_depth, hard_clamp, uncertainty, nodata_value, depth_map,
                          uncertainty_map, gt_depths, reduction='mean'):
    """
    Calculate depth regression loss.

    @param min_depth            Scalar, min depth thresholding parameter for validity check.
    @param hard_clamp           Scalar, max error thresholding parameter for validity check.
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
    invalid_min_depth = depth_map < min_depth  # [B, H_ds*W_ds], behind or too close to camera plane

    # check for additional constraints regarding ground truth scene coordinates
    invalid_gt_distance = depth_reg_error > hard_clamp  # [B, H_ds*W_ds] too far from ground truth scene coordinates

    # combine all constraints
    valid_depth = (invalid_min_depth + invalid_gt_distance + mask_gt_coords_nodata) == 0  # [B, N]

    num_valid_depth = valid_depth.sum(dim=1).cpu().numpy()  # [B]
    num_pixels_batch = valid_depth.numel()
    num_pixels_instance = valid_depth[0].numel()

    # assemble loss
    loss = 0

    """distance loss for pixels whose ground truth is known"""
    if uncertainty is None:
        loss += torch.sum(depth_reg_error * mask_gt_coords_valdata, dim=1)  # [B], applied to all pixels w/ valid labels
    elif uncertainty == 'MLE':
        uncertainty_map = uncertainty_map.view(uncertainty_map.size(0), -1).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        depth_error_square = depth_reg_error.square().clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_unc = 1.0 * torch.log(uncertainty_map) + depth_error_square / (
                2.0 * uncertainty_map.square().clamp(min=1.e-7))  # [B, H_ds*W_ds]
        loss += torch.sum(loss_unc * mask_gt_coords_valdata, dim=1)  # [B], applied to all pixels w/ valid labels

        # diagnosis
        safe_printout(
            'Regression error: depth:  %.2f' % (
                torch.sum(depth_reg_error * mask_gt_coords_valdata).item()
                / max(1, mask_gt_coords_valdata.sum().item())))
    else:
        raise NotImplementedError

    valid_pred_rate = num_valid_depth.sum() / num_pixels_batch  # scalar

    if reduction is None:
        loss /= num_pixels_instance  # [B], each item is the mean over all pixels within one instance
    elif reduction == 'mean':
        loss = loss.sum()  # scalar, mean over each pixels within the batch
        loss /= num_pixels_batch
    else:
        raise NotImplementedError
    return loss, valid_pred_rate
