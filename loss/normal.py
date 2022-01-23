import torch
import numpy as np

from utils.learning import pick_valid_points, xyz2ae, ae2xyz, logits_to_radian
from utils.io import safe_printout


def normal_regression_loss(hard_clamp, uncertainty, nodata_value, normal_logits,
                           uncertainty_map, gt_normals, reduction='mean'):
    """
    Calculate surface normal regression loss. The loss is extracted in the azimuth-elevation mode for stability.

    Reference: Deep Surface Normal Estimation on the 2-Sphere with Confidence Guided Semantic Attention, 2020 ECCV
    Link: https://doi.org/10.1007/978-3-030-58586-0_43

    @param hard_clamp            Scalar hyper-parameter for loss clamp.
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
        safe_printout("normal_logits has {:d} / {:d} nan values!".format(torch.isnan(normal_logits).sum(),
                                                                         normal_logits.numel()))
        safe_printout("normal_reg_error has {:d} / {:d} nan values!".format(torch.isnan(normal_reg_error).sum(),
                                                                            normal_reg_error.numel()))
        safe_printout("azimuth_loss has {:d} / {:d} nan values!".format(torch.isnan(azimuth_loss).sum(),
                                                                        azimuth_loss.numel()))
        safe_printout("elevation_loss has {:d} / {:d} nan values!".format(torch.isnan(elevation_loss).sum(),
                                                                          elevation_loss.numel()))
        safe_printout("pred_normals_ae has {:d} / {:d} nan values!".format(torch.isnan(pred_normals_ae).sum(),
                                                                           pred_normals_ae.numel()))
        safe_printout("gt_normals_ae has {:d} / {:d} nan values!".format(torch.isnan(gt_normals_ae).sum(),
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

    invalid_gt_angle = normal_angle_error > hard_clamp  # [B, H_ds*W_ds] too far from ground truth normals

    """DEBUG"""
    if torch.isnan(normal_angle_error).sum():
        safe_printout("pred_normals_xyz has {:d} / {:d} nan values!".format(torch.isnan(pred_normals_xyz).sum(),
                                                                            pred_normals_xyz.numel()))
        safe_printout("normal_angle_error has {:d} / {:d} nan values!".format(torch.isnan(normal_angle_error).sum(),
                                                                              normal_angle_error.numel()))
        cos_sim = torch.nn.functional.cosine_similarity(pred_normals_xyz, gt_normals_xyz, dim=1)
        safe_printout("cos_sim has {:d} / {:d} nan values!".format(torch.isnan(cos_sim).sum(), cos_sim.numel()))

        arc_cos = torch.acos(cos_sim.clamp(min=-1 + 1.e-7, max=1 - 1.e-7))
        safe_printout("arc_cos has {:d} / {:d} nan values!".format(torch.isnan(arc_cos).sum(), arc_cos.numel()))
    """"""

    # combine all constraints
    valid_normal = (invalid_gt_angle + mask_gt_coords_nodata) == 0  # [B, H_ds*W_ds]
    # invalid_normal = torch.logical_not(valid_normal)  # [B, N]
    num_valid_normal = valid_normal.sum(dim=1).cpu().numpy()  # [B]
    num_pixels_batch = valid_normal.numel()  # B*N
    num_pixels_instance = valid_normal[0].numel()

    # assemble loss
    loss = 0

    """surface normal loss for for pixels whose ground truth is known"""
    if uncertainty is None:
        loss += torch.sum(normal_reg_error * mask_gt_coords_valdata, dim=1)  # [B], applied to all pixels w/ valid labels
    elif uncertainty == 'MLE':
        uncertainty_map = uncertainty_map.view(uncertainty_map.size(0), -1).clamp(min=1.e-7)  # [B, H_ds*W_ds]
        normal_reg_error_square = normal_reg_error.square().clamp(min=1.e-7)  # [B, H_ds*W_ds]
        loss_unc = 2.0 * torch.log(uncertainty_map) + normal_reg_error_square / (
                2.0 * uncertainty_map.square().clamp(min=1.e-7))  # [B, H_ds*W_ds]
        loss += torch.sum(loss_unc * mask_gt_coords_valdata, dim=1)  # [B], applied to all pixels w/ valid labels

        # diagnosis
        safe_printout(
            'Regression error: normal in radian:  %.2f, normal in degree: %.2f' % (
                torch.sum(normal_reg_error * mask_gt_coords_valdata).item()
                / max(1, mask_gt_coords_valdata.sum().item()),
                torch.sum(normal_angle_error * mask_gt_coords_valdata).item()
                / max(1, mask_gt_coords_valdata.sum().item())))
    else:
        raise NotImplementedError

    valid_pred_rate = num_valid_normal.sum() / num_pixels_batch  # scalar

    if reduction is None:
        loss /= num_pixels_instance  # [B], each item is the mean over all pixels within one instance
    elif reduction == 'mean':
        loss = loss.sum()  # scalar, mean over each pixels within the batch
        loss /= num_pixels_batch
    else:
        raise NotImplementedError
    return loss, valid_pred_rate
