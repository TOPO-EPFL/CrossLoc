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
from utils import get_nodata_value

parser = argparse.ArgumentParser(
    description='Test a trained network on a specific scene.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder, e.g. Cambridge_GreatCourt')

parser.add_argument('network', nargs='+', help='file name of a network trained for the scene')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                    help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10,
                    help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                    help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                    help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2],
                    help='test mode: 1 = RGB, 2 = RGB-D')

parser.add_argument('--tiny', '-tiny', action='store_true',
                    help='Load a model with massively reduced capacity for a low memory footprint.')

parser.add_argument('--section', type=str, default='val_drone_sim',
                    help='Dataset to test model performance, could be val or test.')

parser.add_argument('--search_dir', action='store_true',
                    help='Search local directory for all models.')

parser.add_argument('--session', '-sid', default='',
                    help='custom session name appended to output files, useful to separate different runs of a script')

parser.add_argument('--sparse', '-sparse', action='store_true',
                    help='for mode 1 (RGB + ground truth scene coordinates) use sparse scene coordinate initialization targets (eg. for Cambridge) instead of rendered depth maps (eg. for 7scenes and 12scenes).')

parser.add_argument('--save_map', action='store_true',
                    help='Save output coordinate maps for visualization.')

opt = parser.parse_args()

# setup dataset
# if opt.mode < 2: opt.mode = 0 # we do not load ground truth scene coordinates when testing
grayscale = True
if '7scenes' in opt.scene or '12scenes' in opt.scene:
    sparse = False
elif 'Cambridge' in opt.scene or 'epfl' in opt.scene.lower() or 'comballaz' in opt.scene.lower():
    sparse = True
    if 'epfl' in opt.scene.lower() or 'comballaz' in opt.scene.lower():
        grayscale = False
else:
    raise NotImplementedError

if opt.search_dir:
    if isinstance(opt.network, list):
        if len(opt.network) == 1:
            opt.network = opt.network[0]
        else:
            raise NotImplementedError
    if os.path.isdir(opt.network):
        src_dir = opt.network
    else:
        src_dir = os.path.dirname(opt.network)
    src_dir = os.path.abspath(src_dir)
    # opt.network = [os.path.join(src_dir, file) for file in os.listdir(src_dir) if
    #                file.endswith('.pth') or file.endswith('.net')]
    opt.network = [os.path.join(src_dir, dir, 'model.net') for dir in os.listdir(src_dir) if opt.scene in dir and os.path.isdir(os.path.join(src_dir, dir))]
else:
    # local search based on the single input
    if os.path.isdir(opt.network[0]):
        if os.path.exists(os.path.join(opt.network[0], 'model_auto_resume.net')):
            opt.network = [os.path.join(opt.network[0], 'model_auto_resume.net')]
        elif os.path.exists(os.path.join(opt.network[0], 'model.net')):
            opt.network = os.path.join(opt.network[0], 'model.net')
        else:
            raise Exception("No model is found at {:s}!".format(opt.network[0]))

if not isinstance(opt.network, list):
    opt.network = [opt.network]

# all_results_log_path = os.path.join(os.path.dirname(opt.network[0]),
#                                     'all_results_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
all_results_log_path = os.path.join('output/test_log',
                                    'all_results_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
if os.path.exists(all_results_log_path):
    all_results_log = open(all_results_log_path, 'a')
    all_results_log.write('\n')
else:
    all_results_log = open(all_results_log_path, 'w', 1)

opt.network = sorted(opt.network)

for network_in in opt.network:
    # DEBUG
    # if 'baseline' in network_in:
    #     continue

    # load network
    print('Load model from {:s}'.format(network_in))
    if os.path.exists(os.path.abspath(os.path.join(network_in, '../FLAG_training_done.nodata'))):
        print("This model is fully trained.")
    if 'gray' in network_in:
        grayscale_ = True
    else:
        grayscale_ = grayscale
    if opt.section == 'test_real_all':
        testset = CamLocDataset(["./datasets/" + opt.scene + "/" + "val_real",
                                 "./datasets/" + opt.scene + "/" + "test_real"], mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_)
    elif opt.section == "real_all":
        testset = CamLocDataset(["./datasets/" + opt.scene + "/" + "val_real",
                                 "./datasets/" + opt.scene + "/" + "test_real",
                                 "./datasets/" + opt.scene + "/" + "train_real"], mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_)
    elif opt.section == "test_sim_all":
        testset = CamLocDataset(["./datasets/" + opt.scene + "/" + "val_drone_sim",
                                 "./datasets/" + opt.scene + "/" + "val_sim",
                                 "./datasets/" + opt.scene + "/" + "test_drone_sim"], mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_)
    elif opt.section == "sim_all":
        testset = CamLocDataset(["./datasets/" + opt.scene + "/" + "val_drone_sim",
                                 "./datasets/" + opt.scene + "/" + "val_sim",
                                 "./datasets/" + opt.scene + "/" + "test_drone_sim",
                                 "./datasets/" + opt.scene + "/" + "train_sim_aug"], mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_)
    else:
        testset = CamLocDataset("./datasets/" + opt.scene + "/" + opt.section, mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_)
    testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=4)
    if 'unc' in network_in in network_in:
        uncertainty = True
    else:
        uncertainty = False
    if 'comballaz' in opt.scene.lower() or 'epfl' in opt.scene.lower():
        # if 'midrep_' in network_in.lower():
        #     raise NotImplementedError
        # elif 'supercon' in network_in.lower() or 'baseline' in network_in.lower():
        #     network = Network_DGNSC(torch.zeros((3)), opt.tiny, grayscale_, uncertainty)
        # else:
        #     network = Network_DGN(torch.zeros((3)), opt.tiny, grayscale_, uncertainty)
        # network = TransPoseNet(mean, tiny, grayscale, num_task_channel=num_task_channel,
        #                        num_pos_channel=num_pos_channel,
        #                        enc_add_res_block=2, dec_add_res_block=2)

        if '-mlr-' in network_in:
            network = TransPoseNet(torch.zeros(3), opt.tiny, grayscale, num_task_channel=3,
                                   num_pos_channel=1,
                                   enc_add_res_block=2, dec_add_res_block=2, num_mlr=2)
        else:
            network = TransPoseNet(torch.zeros(3), opt.tiny, grayscale, num_task_channel=3,
                                   num_pos_channel=1,
                                   enc_add_res_block=2, dec_add_res_block=2)
    else:
        network = Network(torch.zeros((3)), opt.tiny)

    network.load_state_dict(torch.load(network_in), strict=True)
    network = network.cuda()
    network.eval()

    test_log_path = os.path.join(os.path.dirname(network_in),
                                 'results_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
    if os.path.exists(test_log_path):
        test_log = open(test_log_path, 'a')
        test_log.write('\n')
    else:
        test_log = open(test_log_path, 'w', 1)
    pose_log_path = os.path.join(os.path.dirname(network_in),
                                 'poses_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
    if os.path.exists(pose_log_path):
        pose_log = open(pose_log_path, 'a')
        pose_log.write('\n')
    else:
        pose_log = open(pose_log_path, 'w', 1)

    print('Test images found: ', len(testset))

    # keep track of rotation and translation errors for calculation of the median error
    rErrs = []
    tErrs = []
    avg_time = 0

    pct5 = 0
    pct2 = 0
    pct1 = 0

    nodata_value = get_nodata_value(opt.scene)
    pixel_grid = get_pixel_grid(network.OUTPUT_SUBSAMPLE)
    with torch.no_grad():

        for image, gt_pose, init, focal_length, file in testset_loader:

            focal_length = float(focal_length[0])
            _file = file[0]
            file = file[0].split('/')[-1]  # remove path from file name
            gt_pose = gt_pose[0]
            image = image.cuda()

            ##### plot separated mid-level representation results #####
            # resize and rescale to [-1, 1]
            # image_resize = F.interpolate(image, [256, 256])
            # image_resize = (image_resize - image_resize.min()) / (image_resize.max() - image_resize.min())
            # image_resize = image_resize * 2.0 - 1
            #
            # mid_rep_nm_ls = ['edge_texture', 'keypoints2d', 'resize']
            # for mid_rep_nm in mid_rep_nm_ls:
            # 	if mid_rep_nm != 'resize':
            # 		z = visualpriors.feature_readout(image_resize.cuda(), mid_rep_nm, device='cuda').cpu()
            # 	else:
            # 		z = image_resize.cpu()
            #
            # 	out_folder = "./datasets/" + opt.scene + "/" + opt.section + "/rgb_{:s}".format(mid_rep_nm)
            # 	if not os.path.exists(out_folder):
            # 		os.makedirs(out_folder)
            # 	out_path = os.path.join(out_folder, file)
            #
            # 	TF.to_pil_image(z[0] / 2. + 0.5).save(out_path)
            # continue
            ##########

            ##### cat results from 3 domains #####
            # out_folder = "./datasets/" + opt.scene + "/" + opt.section + "/rgb_cat"
            # if not os.path.exists(out_folder):
            # 	os.makedirs(out_folder)
            # out_path = os.path.join(out_folder, file)
            #
            # fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            # domains = ['real', 'test', 'translate']
            # mid_rep_nm_ls = ['resize', 'edge_texture', 'keypoints2d']
            # for col, domain in enumerate(domains):
            # 	for row, mid_rep_nm in enumerate(mid_rep_nm_ls):
            # 		image_in = os.path.join(os.path.dirname(out_folder), 'rgb_{:s}'.format(mid_rep_nm), file).replace('real', domain)
            # 		assert os.path.exists(image_in)
            # 		image_data = io.imread(image_in)
            # 		# pdb.set_trace()
            # 		axes[row][col].imshow(image_data)
            # 		axes[row][col].axis('off')
            # 		if row == 0:
            # 			axes[row][col].set_title('Domain: {:s}'.format(domain).replace('test', 'synthetic'))
            # 		else:
            # 			axes[row][col].set_title('Style: {:s}'.format(mid_rep_nm))
            # fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
            # plt.close(fig)
            # continue
            ##########

            start_time = time.time()

            # predict scene coordinates and neural guidance
            if 'supercon' in network_in:
                scene_coordinates, _ = network(image, True)
            else:
                scene_coordinates = network(image)  # [1, X, H, W]
            # scene_coordinates = init  # DEBUG
            if uncertainty:
                scene_coords_ = scene_coordinates
                scene_coordinates = scene_coords_[:, :3, :, :]  # [1, 3, H, W]
            if opt.save_map:
                regress_error = torch.norm(init - scene_coordinates.cpu(), p=2, dim=1)  # [1, H, W]
                if uncertainty:
                    prediction_map = torch.cat([scene_coords_[0].cpu(), regress_error], dim=0)  # [4+1, H, W]
                else:
                    prediction_map = torch.cat([scene_coordinates[0], regress_error], dim=0)  # [3+1, H, W]

            out_pose = torch.zeros((4, 4))

            if opt.mode < 2:
                # pose from RGB
                scene_coordinates = scene_coordinates.cpu()
                dsacstar.forward_rgb(
                    scene_coordinates,
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    float(image.size(3) / 2),  # principal point assumed in image center
                    float(image.size(2) / 2),
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE)

            # # use custom pnp solver (off-the-shelf method)
            # tic = time.time()
            # cam_mat = get_cam_mat(image.size(3), image.size(2), focal_length)
            # pixel_grid_crop = pixel_grid[:, 0:scene_coordinates.size(2), 0:scene_coordinates.size(3)].clone()
            # pixel_grid_crop = pixel_grid_crop.view(2, -1)  # [2, H_ds * W_ds]
            # coord_input = scene_coordinates.view(3, -1).transpose(1, 0).contiguous()  # [H_ds*W_ds, 3] tensor
            # pcoord_input = pixel_grid_crop.transpose(1, 0).contiguous()  # [H_ds*W_ds, 3] tensor
            #
            # # pcoord_input, coord_input = pcoord_input.data.cpu().numpy(), coord_input.data.cpu().numpy()
            # # idx_deleted = []
            # # for i, xyz in enumerate(coord_input):
            # # 	if (xyz == nodata_value).any():
            # # 		idx_deleted.append(i)
            # # pcoord_input = np.delete(pcoord_input, idx_deleted, axis=0)
            # # coord_input = np.delete(coord_input, idx_deleted, axis=0)
            # # pcoord_input, coord_input = torch.tensor(pcoord_input), torch.tensor(coord_input)
            #
            # P_out = bpnp(pcoord_input, coord_input, cam_mat, gt_pose, pnp_params).view(-1)
            # P_out_ = P_out.data.cpu().view(-1).numpy()  # [6, ]
            # rot, transl = P_out_[0:3], P_out_[3:6]
            #
            # pose_est = np.eye(4)
            # pose_est[0:3, 0:3] = cv2.Rodrigues(rot)[0].T
            # pose_est[0:3, 3] = -np.dot(pose_est[0:3, 0:3], transl)
            # out_pose = torch.tensor(pose_est)
            # print('Elapsed time %.3f s' % (time.time() - tic))

            else:
                # pose from RGB-D
                dsacstar.forward_rgbd(
                    scene_coordinates,
                    init,  # contains precalculated camera coordinates
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    opt.inlieralpha,
                    opt.maxpixelerror)

            avg_time += time.time() - start_time

            # calculate pose errors
            t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

            gt_R = gt_pose[0:3, 0:3].numpy()
            out_R = out_pose[0:3, 0:3].numpy()

            r_err = np.matmul(out_R, np.transpose(gt_R))
            r_err = cv2.Rodrigues(r_err)[0]
            r_err = np.linalg.norm(r_err) * 180 / math.pi

            print("\nRotation Error: %.2fdeg, Translation Error: %.1fcm" % (r_err, t_err * 100))

            rErrs.append(r_err)
            tErrs.append(t_err * 100)

            if opt.save_map:
                output_info = list([prediction_map.cpu().numpy(), np.array([t_err, r_err])])
                map_dir_nm = 'init-pred-{:s}'.format(os.path.basename(os.path.dirname(network_in)))
                map_path = os.path.abspath(os.path.join(_file, '../../', map_dir_nm))
                os.makedirs(map_path, exist_ok=True)
                np.savez_compressed(os.path.join(map_path, file.replace('.png', '')), a=output_info[0], b=output_info[1])

            if r_err < 5 and t_err < 0.05:
                pct5 += 1
            if r_err < 2 and t_err < 0.02:
                pct2 += 1
            if r_err < 1 and t_err < 0.01:
                pct1 += 1

            # write estimated pose to pose file
            out_pose = out_pose.inverse()

            t = out_pose[0:3, 3]

            # rotation to axis angle
            rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
            angle = np.linalg.norm(rot)
            axis = rot / angle

            # axis angle to quaternion
            q_w = math.cos(angle * 0.5)
            q_xyz = math.sin(angle * 0.5) * axis

            pose_log.write("%s %f %f %f %f %f %f %f %f %f\n" % (
                file,
                q_w, q_xyz[0], q_xyz[1], q_xyz[2],
                t[0], t[1], t[2],
                r_err, t_err))

    median_idx = int(len(rErrs) / 2)
    tErrs.sort()
    rErrs.sort()
    avg_time /= len(rErrs)

    print("\n===================================================")
    print("\nTest complete.")

    print('\nAccuracy:')
    print('\n5cm5deg: %.1f%%' % (pct5 / len(rErrs) * 100))
    print('2cm2deg: %.1f%%' % (pct2 / len(rErrs) * 100))
    print('1cm1deg: %.1f%%' % (pct1 / len(rErrs) * 100))

    print("\nMedian Error: %.1fdeg, %.1fcm" % (rErrs[median_idx], tErrs[median_idx]))
    print("Avg. processing time: %4.1fms" % (avg_time * 1000))
    test_log.write('%f %f %f\n' % (rErrs[median_idx], tErrs[median_idx], avg_time))

    test_log.close()
    pose_log.close()

    all_results_log.write('Separated test log: {:s} \n'.format(test_log_path))
    all_results_log.write('%f %f %f\n' % (rErrs[median_idx], tErrs[median_idx], avg_time))
all_results_log.close()
