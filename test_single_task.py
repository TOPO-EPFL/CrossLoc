import pdb

import torch
import numpy as np
import cv2

import dsacstar

import time
import argparse
import math
import os

from dataloader.dataloader import CamLocDataset
from networks.networks import Network, TransPoseNet
from utils import get_pixel_grid
from legacy_code.train_bpnp import get_pnp_params
from utils import get_nodata_value

from utils import set_random_seed

from typing import Tuple, Union


# all_results_log_path = os.path.join(os.path.dirname(opt.network[0]),
#                                     'all_results_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
# all_results_log_path = os.path.join('output/test_log',
#                                     'all_results_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
# if os.path.exists(all_results_log_path):
#     all_results_log = open(all_results_log_path, 'a')
#     all_results_log.write('\n')
# else:
#     all_results_log = open(all_results_log_path, 'w', 1)
#
# opt.network = sorted(opt.network)
#
# for network_in in opt.network:
#     # DEBUG
#     # if 'baseline' in network_in:
#     #     continue
#
#     # load network
#     print('Load model from {:s}'.format(network_in))
#     if 'gray' in network_in:
#         grayscale_ = True
#     else:
#         grayscale_ = grayscale
#     testset = CamLocDataset("./datasets/" + opt.scene + "/" + opt.section, mode=min(opt.mode, 1),
#                             sparse=sparse, augment=False, grayscale=grayscale_)
#     testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=4)
#     if 'unc' in network_in or 'midrep' in network_in:
#         uncertainty = True
#         if opt.save_map:
#             map_path = "./datasets/" + opt.scene + "/" + opt.section
#             map_path = os.path.join(map_path,
#                                     'uncertainty_map_' + os.path.basename(network_in).replace('.net', '').replace(
#                                         '.pth', ''))
#             if not os.path.exists(map_path):
#                 os.makedirs(map_path)
#     else:
#         uncertainty = False
#         if opt.save_map:
#             map_path = "./datasets/" + opt.scene + "/" + opt.section
#             map_path = os.path.join(map_path,
#                                     'prediction_map_' + os.path.basename(network_in).replace('.net', '').replace('.pth',
#                                                                                                                  ''))
#             if not os.path.exists(map_path):
#                 os.makedirs(map_path)
#     if 'comballaz' in opt.scene.lower() or 'epfl' in opt.scene.lower():
#         # if 'midrep_' in network_in.lower():
#         #     raise NotImplementedError
#         # elif 'supercon' in network_in.lower() or 'baseline' in network_in.lower():
#         #     network = Network_DGNSC(torch.zeros((3)), opt.tiny, grayscale_, uncertainty)
#         # else:
#         #     network = Network_DGN(torch.zeros((3)), opt.tiny, grayscale_, uncertainty)
#         # network = TransPoseNet(mean, tiny, grayscale, num_task_channel=num_task_channel,
#         #                        num_pos_channel=num_pos_channel,
#         #                        enc_add_res_block=2, dec_add_res_block=2)
#         network = TransPoseNet(torch.zeros(3), opt.tiny, grayscale, num_task_channel=3,
#                                num_pos_channel=1,
#                                enc_add_res_block=2, dec_add_res_block=2)
#     else:
#         network = Network(torch.zeros((3)), opt.tiny)
#     network.load_state_dict(torch.load(network_in))
#     network = network.cuda()
#     network.eval()
#
#     test_log_path = os.path.join(os.path.dirname(network_in),
#                                  'results_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
#     if os.path.exists(test_log_path):
#         test_log = open(test_log_path, 'a')
#         test_log.write('\n')
#     else:
#         test_log = open(test_log_path, 'w', 1)
#     pose_log_path = os.path.join(os.path.dirname(network_in),
#                                  'poses_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
#     if os.path.exists(pose_log_path):
#         pose_log = open(pose_log_path, 'a')
#         pose_log.write('\n')
#     else:
#         pose_log = open(pose_log_path, 'w', 1)
#
#     print('Test images found: ', len(testset))
#
#     # keep track of rotation and translation errors for calculation of the median error
#     rErrs = []
#     tErrs = []
#     avg_time = 0
#
#     pct5 = 0
#     pct2 = 0
#     pct1 = 0
#
#     nodata_value = get_nodata_value(opt.scene)
#     pnp_params = get_pnp_params('bpnp', 'bpnp', network.OUTPUT_SUBSAMPLE, 1.0, 100)
#     pixel_grid = get_pixel_grid(network.OUTPUT_SUBSAMPLE)
#     with torch.no_grad():
#
#         for image, gt_pose, init, focal_length, file in testset_loader:
#
#             focal_length = float(focal_length[0])
#             file = file[0].split('/')[-1]  # remove path from file name
#             gt_pose = gt_pose[0]
#             image = image.cuda()
#
#             ##### plot separated mid-level representation results #####
#             # resize and rescale to [-1, 1]
#             # image_resize = F.interpolate(image, [256, 256])
#             # image_resize = (image_resize - image_resize.min()) / (image_resize.max() - image_resize.min())
#             # image_resize = image_resize * 2.0 - 1
#             #
#             # mid_rep_nm_ls = ['edge_texture', 'keypoints2d', 'resize']
#             # for mid_rep_nm in mid_rep_nm_ls:
#             # 	if mid_rep_nm != 'resize':
#             # 		z = visualpriors.feature_readout(image_resize.cuda(), mid_rep_nm, device='cuda').cpu()
#             # 	else:
#             # 		z = image_resize.cpu()
#             #
#             # 	out_folder = "./datasets/" + opt.scene + "/" + opt.section + "/rgb_{:s}".format(mid_rep_nm)
#             # 	if not os.path.exists(out_folder):
#             # 		os.makedirs(out_folder)
#             # 	out_path = os.path.join(out_folder, file)
#             #
#             # 	TF.to_pil_image(z[0] / 2. + 0.5).save(out_path)
#             # continue
#             ##########
#
#             ##### cat results from 3 domains #####
#             # out_folder = "./datasets/" + opt.scene + "/" + opt.section + "/rgb_cat"
#             # if not os.path.exists(out_folder):
#             # 	os.makedirs(out_folder)
#             # out_path = os.path.join(out_folder, file)
#             #
#             # fig, axes = plt.subplots(3, 3, figsize=(9, 9))
#             # domains = ['real', 'test', 'translate']
#             # mid_rep_nm_ls = ['resize', 'edge_texture', 'keypoints2d']
#             # for col, domain in enumerate(domains):
#             # 	for row, mid_rep_nm in enumerate(mid_rep_nm_ls):
#             # 		image_in = os.path.join(os.path.dirname(out_folder), 'rgb_{:s}'.format(mid_rep_nm), file).replace('real', domain)
#             # 		assert os.path.exists(image_in)
#             # 		image_data = io.imread(image_in)
#             # 		# pdb.set_trace()
#             # 		axes[row][col].imshow(image_data)
#             # 		axes[row][col].axis('off')
#             # 		if row == 0:
#             # 			axes[row][col].set_title('Domain: {:s}'.format(domain).replace('test', 'synthetic'))
#             # 		else:
#             # 			axes[row][col].set_title('Style: {:s}'.format(mid_rep_nm))
#             # fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
#             # plt.close(fig)
#             # continue
#             ##########
#
#             start_time = time.time()
#
#             # predict scene coordinates and neural guidance
#             if 'supercon' in network_in:
#                 scene_coordinates, _ = network(image, True)
#             else:
#                 scene_coordinates = network(image)
#             # scene_coordinates = init  # DEBUG
#             if uncertainty:
#                 scene_coords_ = scene_coordinates
#                 scene_coordinates = scene_coords_[:, :3, :, :]  # [1, 3, H, W]
#             if opt.save_map:
#                 regress_error = torch.norm(init - scene_coordinates.cpu(), p=2, dim=1)
#                 if uncertainty:
#                     prediction_map = scene_coords_[:, -1, :, :].cpu()  # [1, H, W]
#                     prediction_map = torch.cat([prediction_map, regress_error], dim=0)
#                 else:
#                     prediction_map = regress_error
#
#             out_pose = torch.zeros((4, 4))
#
#             if opt.mode < 2:
#                 # pose from RGB
#                 scene_coordinates = scene_coordinates.cpu()
#                 dsacstar.forward_rgb(
#                     scene_coordinates,
#                     out_pose,
#                     opt.hypotheses,
#                     opt.threshold,
#                     focal_length,
#                     float(image.size(3) / 2),  # principal point assumed in image center
#                     float(image.size(2) / 2),
#                     opt.inlieralpha,
#                     opt.maxpixelerror,
#                     network.OUTPUT_SUBSAMPLE)
#
#             # # use custom pnp solver (off-the-shelf method)
#             # tic = time.time()
#             # cam_mat = get_cam_mat(image.size(3), image.size(2), focal_length)
#             # pixel_grid_crop = pixel_grid[:, 0:scene_coordinates.size(2), 0:scene_coordinates.size(3)].clone()
#             # pixel_grid_crop = pixel_grid_crop.view(2, -1)  # [2, H_ds * W_ds]
#             # coord_input = scene_coordinates.view(3, -1).transpose(1, 0).contiguous()  # [H_ds*W_ds, 3] tensor
#             # pcoord_input = pixel_grid_crop.transpose(1, 0).contiguous()  # [H_ds*W_ds, 3] tensor
#             #
#             # # pcoord_input, coord_input = pcoord_input.data.cpu().numpy(), coord_input.data.cpu().numpy()
#             # # idx_deleted = []
#             # # for i, xyz in enumerate(coord_input):
#             # # 	if (xyz == nodata_value).any():
#             # # 		idx_deleted.append(i)
#             # # pcoord_input = np.delete(pcoord_input, idx_deleted, axis=0)
#             # # coord_input = np.delete(coord_input, idx_deleted, axis=0)
#             # # pcoord_input, coord_input = torch.tensor(pcoord_input), torch.tensor(coord_input)
#             #
#             # P_out = bpnp(pcoord_input, coord_input, cam_mat, gt_pose, pnp_params).view(-1)
#             # P_out_ = P_out.data.cpu().view(-1).numpy()  # [6, ]
#             # rot, transl = P_out_[0:3], P_out_[3:6]
#             #
#             # pose_est = np.eye(4)
#             # pose_est[0:3, 0:3] = cv2.Rodrigues(rot)[0].T
#             # pose_est[0:3, 3] = -np.dot(pose_est[0:3, 0:3], transl)
#             # out_pose = torch.tensor(pose_est)
#             # print('Elapsed time %.3f s' % (time.time() - tic))
#
#             else:
#                 # pose from RGB-D
#                 dsacstar.forward_rgbd(
#                     scene_coordinates,
#                     init,  # contains precalculated camera coordinates
#                     out_pose,
#                     opt.hypotheses,
#                     opt.threshold,
#                     opt.inlieralpha,
#                     opt.maxpixelerror)
#
#             avg_time += time.time() - start_time
#
#             # calculate pose errors
#             t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))
#
#             gt_R = gt_pose[0:3, 0:3].numpy()
#             out_R = out_pose[0:3, 0:3].numpy()
#
#             r_err = np.matmul(out_R, np.transpose(gt_R))
#             r_err = cv2.Rodrigues(r_err)[0]
#             r_err = np.linalg.norm(r_err) * 180 / math.pi
#
#             print("\nRotation Error: %.2fdeg, Translation Error: %.1fcm" % (r_err, t_err * 100))
#
#             rErrs.append(r_err)
#             tErrs.append(t_err * 100)
#
#             if opt.save_map:
#                 output_info = list([prediction_map.cpu().numpy(), np.array([t_err, r_err])])
#                 np.savez_compressed(os.path.join(map_path, file.replace('.png', '')), a=output_info[0],
#                                     b=output_info[1])
#
#             if r_err < 5 and t_err < 0.05:
#                 pct5 += 1
#             if r_err < 2 and t_err < 0.02:
#                 pct2 += 1
#             if r_err < 1 and t_err < 0.01:
#                 pct1 += 1
#
#             # write estimated pose to pose file
#             out_pose = out_pose.inverse()
#
#             t = out_pose[0:3, 3]
#
#             # rotation to axis angle
#             rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
#             angle = np.linalg.norm(rot)
#             axis = rot / angle
#
#             # axis angle to quaternion
#             q_w = math.cos(angle * 0.5)
#             q_xyz = math.sin(angle * 0.5) * axis
#
#             pose_log.write("%s %f %f %f %f %f %f %f %f %f\n" % (
#                 file,
#                 q_w, q_xyz[0], q_xyz[1], q_xyz[2],
#                 t[0], t[1], t[2],
#                 r_err, t_err))
#
#     median_idx = int(len(rErrs) / 2)
#     tErrs.sort()
#     rErrs.sort()
#     avg_time /= len(rErrs)
#
#     print("\n===================================================")
#     print("\nTest complete.")
#
#     print('\nAccuracy:')
#     print('\n5cm5deg: %.1f%%' % (pct5 / len(rErrs) * 100))
#     print('2cm2deg: %.1f%%' % (pct2 / len(rErrs) * 100))
#     print('1cm1deg: %.1f%%' % (pct1 / len(rErrs) * 100))
#
#     print("\nMedian Error: %.1fdeg, %.1fcm" % (rErrs[median_idx], tErrs[median_idx]))
#     print("Avg. processing time: %4.1fms" % (avg_time * 1000))
#     test_log.write('%f %f %f\n' % (rErrs[median_idx], tErrs[median_idx], avg_time))
#
#     test_log.close()
#     pose_log.close()
#
#     all_results_log.write('Separated test log: {:s} \n'.format(test_log_path))
#     all_results_log.write('%f %f %f\n' % (rErrs[median_idx], tErrs[median_idx], avg_time))
# all_results_log.close()


def _config_parser():
    """
    Task specific argument parser
    """
    parser = argparse.ArgumentParser(
        description='Initialize a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    """General training parameter"""
    # Dataset and dataloader
    parser.add_argument('scene', help='name of a scene in the dataset folder')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size for baseline dataloader, NOT applicable to supervised contrastive learning.')

    parser.add_argument('--grayscale', '-grayscale', action='store_true',
                        help='use grayscale image as model input')

    parser.add_argument('--translated', action='store_true',
                        help='to use translated training images (derived from REAL data points)')

    parser.add_argument('--synthetic_only', action='store_true',
                        help='to use synthetic data only, contrastive learning is forced to be turned off')

    parser.add_argument('--task', type=str, required=True,
                        help='specify the single regression task, should be "coord", "depth" or "normal"')

    # Network structure
    parser.add_argument('--network_in', type=str, default=None,
                        help='file name of a network initialized for the scene')

    parser.add_argument('--tiny', '-tiny', action='store_true',
                        help='train a model with massively reduced capacity for a low memory footprint.')

    """I/O parameters"""
    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, useful to separate different runs of a script')

    parser.add_argument('--ckpt_dir', type=str, default='',
                        help="directory to save checkpoint models.")

    parser.add_argument('--search_dir', action='store_true',
                        help='Search local directory for all models.')

    parser.add_argument('--keywords', default=None, nargs='+',
                        help="Keywords to filter out some network weight paths.")

    """DSAC* PnP solver parameters"""
    # Default values are used
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

    """Regression validity check"""
    parser.add_argument('--tolerance', '-tol', type=float, default=50.0,
                        help="A list of tolerance values to determine valid regression prediction.")

    """Uncertainty loss parameter"""
    parser.add_argument('--uncertainty', '-uncertainty', action='store_true',
                        help='enable uncertainty learning')

    opt = parser.parse_args()

    return opt


def _config_weight_path(network_in: Union[str, list], keywords: Union[None, str, list] = None,
                        search_dir: bool = False) -> list:
    """
    Identify the paths to the model weights.
    @param network_in:      One or multiple parental directories or paths to network model weight.
    @param keywords:        One or multiple (union) keywords to search for.
    @param search_dir:      Search for all model weights found in the single specified directory.
    @return: network_paths  One or multiple paths to the network weights filtered by the keywords.
    """

    if isinstance(network_in, list):
        # A list of model weights or directories containing model weights
        _network_in = sorted([os.path.abspath(item) for item in network_in])
    elif isinstance(network_in, str):
        # A path to a model weight of a directory containing model weight
        _network_in = [os.path.abspath(network_in)]
    else:
        raise NotImplementedError

    if search_dir:
        # Turn the directory into a list of sub-directories
        assert len(_network_in) == 1, "_network_in must have one item in search_dir mode. Now it's: {}".format(
            _network_in)
        assert os.path.isdir(_network_in[0]), "_network_in {:s} is not a directory.".format(_network_in[0])
        src_dir = os.path.abspath(_network_in[0])
        _network_in = [os.path.join(src_dir, sub_dir) for sub_dir in os.listdir(src_dir)]

    """Get the exact model weights path"""
    network_paths = []
    for path in _network_in:
        assert os.path.exists(path), "Network input path {:s} is not found.".format(path)
        if os.path.isdir(path):
            # some directory containing weight
            model_path = os.path.join(path, 'model.net')
            if os.path.exists(model_path):
                network_paths.append(model_path)
        elif os.path.isfile(path):
            # some designated model path
            if path.endswith("model.net"):
                network_paths.append(path)
        else:
            raise NotImplementedError

    """Keyword search"""
    if keywords is None:
        pass
    elif isinstance(keywords, str):
        keywords = [keywords]
    elif isinstance(keywords, list):
        pass
    else:
        raise NotImplementedError

    if keywords is not None:
        network_paths_raw = network_paths.copy()
        network_paths = []
        for path in network_paths_raw:
            flags = np.array([keyword in os.path.dirname(path) for keyword in keywords])
            if flags.any():
                network_paths.append(path)
        network_paths = np.sort(np.unique(network_paths)).tolist()

    print("With the keywords {:},".format(keywords), end=" ")
    print("the following {:d} network weight paths are retrieved:".format(len(network_paths)))
    for idx, path in enumerate(network_paths):
        print("Network weight #{:d}: {:s}".format(idx, path))

    return network_paths


def _config_dataloader(scene, task, grayscale, batch_size, nodata_value, section='test'):
    """
    Configure dataloader (task specific).
    """
    if 'epfl' in scene.lower() or 'comballaz' in scene.lower():
        pass
    else:
        raise NotImplementedError

    assert section in ['train', 'val', 'test', 'train_aug', 'test_aug'], "section {} is not supported.".format(section)

    # original dataset to calculate mean
    root_sim = "./datasets/" + scene + "/train_sim_aug"
    root_real = "./datasets/" + scene + "/train_translated" if translated else "./datasets/" + scene + "/train_real"
    trainset_vanilla = CamLocDataset([root_sim, root_real], coord=True, depth=True, normal=True,
                                     augment=False, raw_image=False, mute=True)
    trainset_loader_vanilla = torch.utils.data.DataLoader(trainset_vanilla, shuffle=False, batch_size=1,
                                                          num_workers=mp.cpu_count() // 2, pin_memory=True,
                                                          collate_fn=trainset_vanilla.batch_resize)

    flag_coord = task == 'coord'
    flag_depth = task == 'depth'
    flag_normal = task == 'normal'

    if synthetic_only:
        trainset = CamLocDataset(root_sim, coord=flag_coord, depth=flag_depth, normal=flag_normal,
                                 augment=True, grayscale=grayscale, raw_image=False)
        trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                      num_workers=mp.cpu_count() // 2,
                                                      pin_memory=True, collate_fn=trainset.batch_resize)
        logging.info("Warning: this training uses synthetic data only. {:d} iterations per epoch.".format(len(trainset)))
    else:
        trainset = CamLocDataset([root_sim, root_real], coord=flag_coord, depth=flag_depth, normal=flag_normal,
                                 augment=True, grayscale=grayscale, raw_image=False)
        trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                      num_workers=mp.cpu_count() // 2,
                                                      pin_memory=True, collate_fn=trainset.batch_resize)
        logging.info("This training uses vanilla mode (mixing sim & real data naively). {:d} iterations per epoch".format(
            len(trainset)))

    return trainset, trainset_loader, mean


def main():
    """
    Main function.
    """

    """Initialization"""
    set_random_seed(2021)
    opt = _config_parser()

    nodata_value = get_nodata_value(opt.scene)

    network_paths = _config_weight_path(opt.network_in, opt.keywords, opt.search_dir)

    from train_single_task import _config_dataloader
    trainset, trainset_loader, mean = _config_dataloader(opt.scene, opt.task, opt.translated, opt.grayscale,
                                                         opt.synthetic_only,
                                                         0, 0, 0, 0, 0,
                                                         opt.batch_size, nodata_value)




if __name__ == "__main__":
    main()
