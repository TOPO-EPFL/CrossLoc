import logging
import os
import pdb

import cv2

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp

import dsacstar
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from dataloader.dataloader import CamLocDataset
from networks.networks import TransPoseNet, Network
from utils.learning import pick_valid_points, logits_to_radian, ae2xyz


def config_dataloader(scene, task, grayscale, section_keyword, fullsize, mute=False):
    """
    Configure evaluation dataloader.
    """
    if 'urbanscape' in scene.lower() or 'naturescape' in scene.lower():
        pass
    else:
        raise NotImplementedError

    # fullsize adaptive tweaking
    if task == 'semantics':
        _scene = scene
        assert fullsize
    else:
        _scene = scene + '-fullsize' if fullsize else scene

    data_to_load = "./datasets/" + scene + "/" + section_keyword

    if os.path.exists(data_to_load):
        if mute:
            pass
        else:
            print("Loading evaluation data at {:s}".format(data_to_load))
    else:
        print("Loading special section {:s}".format(section_keyword))
        if section_keyword == 'test_real_all':
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_real",
                            "./datasets/" + scene + "/" + "test_drone_real"]
        elif section_keyword == "real_all":
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_real",
                            "./datasets/" + scene + "/" + "test_drone_real",
                            "./datasets/" + scene + "/" + "train_drone_real"]
        elif section_keyword == "test_sim_all":
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_sim",
                            "./datasets/" + scene + "/" + "val_sim",
                            "./datasets/" + scene + "/" + "test_drone_sim"]
        elif section_keyword == "sim_all":
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_sim",
                            "./datasets/" + scene + "/" + "val_sim",
                            "./datasets/" + scene + "/" + "test_drone_sim",
                            "./datasets/" + scene + "/" + "train_sim"]
        else:
            raise NotImplementedError

    flag_coord = task == 'coord'
    flag_depth = task == 'depth'
    flag_normal = task == 'normal'
    flag_semantics = task == 'semantics'

    batch_size = 1 if flag_coord else 4
    eval_set = CamLocDataset(data_to_load, coord=flag_coord, depth=flag_depth, normal=flag_normal,
                             semantics=flag_semantics, mute=mute,
                             augment=False, grayscale=grayscale, raw_image=True, fullsize=fullsize)
    eval_set_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=min(mp.cpu_count() // 2, 6),
                                                  pin_memory=True)
    logging.info("This evaluation dataloader has {:d} data points in total.".format(len(eval_set)))

    return eval_set, eval_set_loader


def config_network(scene, task, tiny, grayscale, uncertainty, fullsize, network_in, num_enc=0):
    """
    Configure evaluation network.
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
        network = TransPoseNet(torch.zeros(num_task_channel), tiny, grayscale, num_task_channel=num_task_channel,
                               num_pos_channel=num_pos_channel,
                               enc_add_res_block=2, dec_add_res_block=2, full_size_output=fullsize,
                               num_mlr=num_enc)
    else:
        network = Network(torch.zeros(3), tiny)

    network.load_state_dict(torch.load(network_in), strict=True)
    logging.info("Successfully loaded %s." % network_in)
    network = network.cuda()
    network.eval()

    return network


def get_pose_err(gt_pose: np.ndarray, est_pose: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation and rotation error between two 4x4 transformation matrices.
    """
    transl_err = np.linalg.norm(gt_pose[0:3, 3] - est_pose[0:3, 3])

    rot_err = est_pose[0:3, 0:3].T.dot(gt_pose[0:3, 0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1, 3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.
    rot_err = rot_err[0]
    return transl_err, rot_err


def scene_coords_eval(scene_coords, gt_coords, gt_pose, nodata_value, focal_length, image_h, image_w,
                      hypotheses, threshold, inlier_alpha, max_pixel_error, output_subsample) \
        -> Tuple[float, float, list, list, torch.Tensor]:
    """
    Evaluate predicted scene coordinates. Batch size must be one.
    DSAC* PnP solver is adopted. Code reference: https://github.com/vislearn/dsacstar.
    @param scene_coords             [1, 3, H, W], predicted scene coordinates.
    @param gt_coords                [1, 3, H, W], ground-truth scene coordinates.
    @param gt_pose                  [1, 4, 4] cam-to-world matrix.
    @param nodata_value             Nodata value.
    @param focal_length             Camera focal length.
    @param image_h                  Image height.
    @param image_w                  Image width.
    @param hypotheses               DSAC* PnP solver parameter.
    @param threshold                DSAC* PnP solver parameter.
    @param inlier_alpha             DSAC* PnP solver parameter.
    @param max_pixel_error          DSAC* PnP solver parameter.
    @param output_subsample         DSAC* PnP solver parameter.

    @return: t_err, r_err, est_xyz, coords_error for has-data pixels, and 4x4 out_pose
    """
    gt_pose = gt_pose[0].cpu()

    """metrics on camera pose"""
    # compute 6D camera pose
    out_pose = torch.zeros((4, 4))
    scene_coords = scene_coords.cpu()
    dsacstar.forward_rgb(
        scene_coords,
        out_pose,
        hypotheses,
        threshold,
        focal_length,
        float(image_w / 2),  # principal point assumed in image center
        float(image_h / 2),
        inlier_alpha,
        max_pixel_error,
        output_subsample)

    # calculate pose error
    t_err, r_err = get_pose_err(gt_pose.numpy(), out_pose.numpy())

    # estimated XYZ position
    est_xyz = out_pose[0:3, 3].tolist()

    """metrics on regression error"""
    scene_coords = scene_coords.view(scene_coords.size(0), 3, -1)  # [1, 3, H*W]
    gt_coords = gt_coords.view(gt_coords.size(0), 3, -1)  # [1, 3, H*W]
    mask_gt_coords_valdata = pick_valid_points(gt_coords, nodata_value, boolean=True)  # [1, H*W]

    coords_error = torch.norm(gt_coords - scene_coords, dim=1, p=2)  # [1, H*W]
    coords_error_valdata = coords_error[mask_gt_coords_valdata].tolist()  # [X]

    print("\nRotation Error: %.2f deg, Translation Error: %.1f m, Mean coord prediction error: %.1f m" % (
        r_err, t_err, np.mean(coords_error_valdata)))
    return t_err, r_err, est_xyz, coords_error_valdata, out_pose.clone()


def scene_coords_printout(t_err_ls, r_err_ls, est_xyz_ls, coords_error_ls, testing_log,
                          network_path, section, file_name_ls) -> None:
    """
    Save the scene coordinate evaluation results to file.
    @param t_err_ls:         A list of translation errors.
    @param r_err_ls:         A list of rotation errors.
    @param est_xyz_ls:       A list of predicted coordinates.
    @param coords_error_ls:  A list of coordinate prediction errors.
    @param testing_log:      String for testing log path.
    @param network_path:     String for the network weight path.
    @param section:          String for the evaluation set.
    @param file_name_ls:     A list of the files in the evaluation set.
    @return:
    """
    t_err_ls = np.array(t_err_ls)  # [N]
    r_err_ls = np.array(r_err_ls)  # [N]
    est_xyz_ls = np.stack(est_xyz_ls, axis=0)  # [N, 3]
    coords_error_ls = np.concatenate(coords_error_ls)  # [X]

    pct30_10 = np.sum(np.logical_and(t_err_ls < 30.0, r_err_ls < 10.0))
    pct20_10 = np.sum(np.logical_and(t_err_ls < 20.0, r_err_ls < 10.0))
    pct10_10 = np.sum(np.logical_and(t_err_ls < 10.0, r_err_ls < 10.0))
    pct10_7 = np.sum(np.logical_and(t_err_ls < 10.0, r_err_ls < 7.0))
    pct5 = np.sum(np.logical_and(t_err_ls < 5.0, r_err_ls < 5.0))
    pct3 = np.sum(np.logical_and(t_err_ls < 3.0, r_err_ls < 3.0))

    eval_str = '\nAccuracy:'
    eval_str += '\n30m10deg: %.1f%%\n20m10deg: %.1f%%' % (pct30_10 / len(t_err_ls) * 100,
                                                          pct20_10 / len(t_err_ls) * 100)
    eval_str += '\n10m7deg: %.1f%%' % (pct10_7 / len(t_err_ls) * 100)
    eval_str += '\n10m10deg: %.1f%%' % (pct10_10 / len(t_err_ls) * 100) + '\n5m5deg: %.1f%%' % (
            pct5 / len(t_err_ls) * 100)
    eval_str += '\n3m3deg: %.1f%%' % (pct3 / len(t_err_ls) * 100)
    eval_str += "\nMedian Error: %.1f deg, %.2f m" % (np.median(r_err_ls), np.median(t_err_ls))
    eval_str += "\nMean Errors: %.1f plus-minus %.1f deg, %.2f plus-minus %.2f m" % (
        np.mean(r_err_ls), np.std(r_err_ls), np.mean(t_err_ls), np.std(t_err_ls))
    eval_str += "\nCoordinate regression error: mean {:.1f}, std {:.1f}, median {:.1f}".format(
        np.mean(coords_error_ls), np.std(coords_error_ls), np.median(coords_error_ls))
    print(eval_str)

    with open(testing_log, 'a') as f:
        f.write("{:s} Evaluation on section {:s} {:s}".format('=' * 20, section, '=' * 20) + '\n')
        f.write(eval_str)
        f.write('\n')

    np.save(os.path.join(os.path.dirname(network_path), '{:s}_{:s}_out_xyz_poses.npy'.format(
        section, os.path.basename(network_path))), est_xyz_ls)

    with open(os.path.join(os.path.dirname(network_path), '{:s}_{:s}_out_xyz_poses.txt'.format(
            section, os.path.basename(network_path))), 'w') as f:
        for file, pose_xyz in zip(file_name_ls, est_xyz_ls):
            f.write(file + ' {:.2f} {:.2f} {:.2f}'.format(*pose_xyz) + '\n')


def depth_eval(depth, gt_depth, nodata_value) -> Tuple[float, float]:
    """
    Evaluate the predicted depth.
    @param depth             [B, 1, H, W], predicted scene coordinates.
    @param gt_depth          [B, 1, H, W], ground-truth scene coordinates.
    @param nodata_value      Nodata value.

    @return: depth_abs_rel, depth_rms metrics based on has-data pixels
    """

    depth = depth.view(depth.size(0), -1).cpu()  # [B, H*W]
    gt_depth = gt_depth.view(depth.size(0), -1).cpu()  # [B, H*W]
    depth_reg_error = torch.abs(depth - gt_depth)  # [B, H*W]

    # check for invalid/unknown ground truth scene coordinates
    mask_gt_coords_valdata = pick_valid_points(gt_depth.unsqueeze(1), nodata_value, boolean=True)  # [B, H*W]

    depth_abs_rel = (depth_reg_error * mask_gt_coords_valdata / gt_depth).sum() / mask_gt_coords_valdata.sum()  # scalar
    depth_rms = ((depth_reg_error * mask_gt_coords_valdata).square().sum() / mask_gt_coords_valdata.sum()).sqrt()  # scalar

    return depth_abs_rel, depth_rms


def depth_printout(depth_abs_rel_ls, depth_rms_ls, testing_log, section) -> None:
    """
    Save the depth prediction results to file.
    @param depth_abs_rel_ls:    A list of depth absolute relative error.
    @param depth_rms_ls:        A list of RMS error.
    @param testing_log:         String for testing log path.
    @param section:             String for the evaluation set.
    """

    depth_abs_rel_ls = np.array(depth_abs_rel_ls)  # [N]
    depth_rms_ls = np.array(depth_rms_ls)  # [N]

    eval_str = "Depth accuracy:"
    eval_str += "\nabsolute relative error, mean: {:.2f}%, median: {:.2f}%".format(
        np.mean(depth_abs_rel_ls) * 100.0, np.median(depth_abs_rel_ls) * 100.0)
    eval_str += "\nRMS error, mean: {:.2f}m, median: {:.2f}m".format(np.mean(depth_rms_ls), np.median(depth_rms_ls))

    print(eval_str)
    with open(testing_log, 'a') as f:
        f.write("{:s} Evaluation on section {:s} {:s}".format('=' * 20, section, '=' * 20) + '\n')
        f.write(eval_str)
        f.write('\n')


def normal_eval(normal_logits, gt_normals, nodata_value) -> float:
    """
    Evaluate the surface normal vector prediction.
    @param normal_logits        [B, 2, H_ds, W_ds] Predicted surface normal logits tensor.
    @param gt_normals           [B, 3, H_ds, W_ds] ---> [B, 3, 60, 80] by default w/o augmentation
    @param nodata_value         Scalar to indicate NODATA element of ground truth scene coordinates.
    @return:                    Average surface normal regression error in degree.
    """

    batch_size = normal_logits.size(0)
    normal_logits = normal_logits.view(batch_size, 2, -1).cpu()  # [B, 2, H_ds*W_ds]
    gt_normals = gt_normals.view(batch_size, 3, -1).cpu()  # [B, 3, H_ds*W_ds]

    pred_normals_ae = logits_to_radian(normal_logits)  # [B, 2, H_ds*W_ds], range [-pi, pi]
    pred_normals_xyz = ae2xyz(pred_normals_ae.clone().detach())  # [B, 3, H_ds*W_ds]

    normal_angle_error = torch.nn.functional.cosine_similarity(pred_normals_xyz, gt_normals, dim=1)  # [B, H_ds*W_ds], range [-1, 1]
    normal_angle_error = torch.acos(normal_angle_error.clamp(min=-1 + 1.e-7, max=1 - 1.e-7))  # [B, H_ds*W_ds], range [0, pi]
    normal_angle_error = normal_angle_error / np.pi * 180.0  # range [0, 180], unit in degrees

    mask_gt_coords_valdata = pick_valid_points(gt_normals, nodata_value, boolean=True)  # [B, H_ds*W_ds]
    normal_reg_error = (normal_angle_error * mask_gt_coords_valdata).sum() / mask_gt_coords_valdata.sum()
    return normal_reg_error


def normal_printout(normal_angular_err_ls, testing_log, section) -> None:
    """
    Save the surface normal prediction results to file.
    @param normal_angular_err_ls:   A list of surface normal regression error.
    @param testing_log:             String for testing log path.
    @param section:                 String for the evaluation set.
    """
    normal_angular_err_ls = np.array(normal_angular_err_ls)  # [N]

    eval_str = "Surface normal accuracy:"
    eval_str += "\nangular prediction error, mean: {:.1f} deg, median: {:.1f} deg".format(
        np.mean(normal_angular_err_ls), np.median(normal_angular_err_ls))

    print(eval_str)
    with open(testing_log, 'a') as f:
        f.write("{:s} Evaluation on section {:s} {:s}".format('=' * 20, section, '=' * 20) + '\n')
        f.write(eval_str)
        f.write('\n')


class SemanticsEvaluator(object):
    """
    Helper to evaluate semantics segmentation performance.
    Reference: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def semantic_eval(semantic_logits, gt_label, mute=False):
    """
    Evaluate semantics segmentation result. The size of logits and label are the same as the raw image.
    @param semantic_logits:     [B, 6, H, W]
    @param gt_label:            [B, 1, H, W]
    @param mute:                flag
    @return:
    """

    gt_label = gt_label.squeeze(1)  # [B, H, W]
    class_prediction = torch.argmax(F.log_softmax(semantic_logits, dim=1), dim=1).cpu()  # [B, H, W]
    assert gt_label.shape == class_prediction.shape

    miou_ls, fwiou_ls, acc_ls = [], [], []  # [B]
    evaluator = SemanticsEvaluator(6)
    for this_gt_label, this_class_pred in zip(gt_label.cpu().numpy(), class_prediction.cpu().numpy()):
        evaluator.reset()
        evaluator.add_batch(this_gt_label, this_class_pred)
        miou_ls.append(evaluator.Mean_Intersection_over_Union())
        fwiou_ls.append(evaluator.Frequency_Weighted_Intersection_over_Union())
        acc_ls.append(evaluator.Pixel_Accuracy())
    miou_ls, fwiou_ls, acc_ls = np.array(miou_ls), np.array(fwiou_ls), np.array(acc_ls)
    if not mute:
        print("Metrics within the batch: mean accuracy: {:.2f}%, mean IoU: {:.2f}%, frequency weighted IoU: {:.2f}%".
            format(acc_ls.mean() * 100, miou_ls.mean() * 100, fwiou_ls.mean() * 100))

    return class_prediction, miou_ls, fwiou_ls, acc_ls


def semantic_plotter(image, class_prediction, gt_label, network_path, section) -> None:
    """
    Plot the semantic evaluation result for sanity check.
    @param image:               [B, 3, H, W] RGB images.
    @param class_prediction:    [B, H, W] predicted classes
    @param gt_label:            [B, 1, H, W] ground-truth classes.
    @param network_path:        String for network weight path.
    @param section:             Evaluation set name.
    @return:
    """
    batch_size = image.size(0)  # batch size
    fig, axes = plt.subplots(batch_size, 3, figsize=(3, batch_size))
    for row in range(batch_size):

        axes[row, 0].axis('off')
        axes[row, 0].imshow(image[row].numpy().transpose(1, 2, 0))

        axes[row, 1].axis('off')
        axes[row, 1].imshow(class_prediction[row], vmin=0, vmax=6)

        axes[row, 2].axis('off')
        axes[row, 2].imshow(gt_label[row][0], vmin=0, vmax=6)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(os.path.abspath(os.path.join(
        network_path, '..', 'sm_section_{:s}_batch_{:d}'.format(section, j))),
        bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)


def semantic_printout(accuracy_ls, mean_iou_ls, fw_iou_ls, testing_log, section) -> None:
    """
    Save the semantic segmentation evaluation results to file.
    @param accuracy_ls:
    @param mean_iou_ls:
    @param fw_iou_ls:
    @param testing_log:      String for testing log path.
    @param section:          String for the evaluation set.
    @return:
    """
    accuracy_ls = np.concatenate(accuracy_ls)
    mean_iou_ls = np.concatenate(mean_iou_ls)
    fw_iou_ls = np.concatenate(fw_iou_ls)

    accuracy_str = "Pixel accuracy, mean: {:.2f}, median: {:.2f}".format(
        np.mean(accuracy_ls) * 100, np.median(accuracy_ls) * 100)
    print(accuracy_str)

    mean_iou_str = "Mean IoU, mean: {:.2f}, median: {:.2f}".format(
        np.mean(mean_iou_ls) * 100, np.median(mean_iou_ls) * 100)
    print(mean_iou_str)

    fw_iou_str = "Frequency weighted IoU, mean: {:.2f}, median: {:.2f}".format(
        np.mean(fw_iou_ls) * 100, np.median(fw_iou_ls) * 100)
    print(fw_iou_str)

    # out_path = os.path.join(os.path.dirname(network_path), 'results_{:s}_{:s}_section_{:s}.txt'.format(
    #     os.path.basename(network_path), task, this_section))
    # with open(out_path, 'w') as f:
    #     f.write(accuracy_str + '\n')
    #     f.write(mean_iou_str + '\n')

    with open(testing_log, 'a') as f:
        f.write("{:s} Evaluation on section {:s} {:s}".format('=' * 20, section, '=' * 20) + '\n')
        f.write(accuracy_str + '\n')
        f.write(mean_iou_str + '\n')
        f.write(fw_iou_str + '\n')
        f.write('\n')

