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
from utils import get_nodata_value, pick_valid_points

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
    opt.network = [os.path.join(src_dir, dir, 'model.net') for dir in os.listdir(src_dir) if opt.scene in dir and 'depth' in dir and os.path.isdir(os.path.join(src_dir, dir))]
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

all_results_log_path = os.path.join('output/test_log',
                                    'all_results_depth_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
if os.path.exists(all_results_log_path):
    all_results_log = open(all_results_log_path, 'a')
    all_results_log.write('\n')
else:
    all_results_log = open(all_results_log_path, 'w', 1)

opt.network = sorted(opt.network)

print("The following {:d} models are to be processed:".format(len(opt.network)))
for network_in in opt.network:
    print(network_in)

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
                                sparse=sparse, augment=False, grayscale=grayscale_,
                                coord=False, depth=True, normal=False)
    elif opt.section == "real_all":
        testset = CamLocDataset(["./datasets/" + opt.scene + "/" + "val_real",
                                 "./datasets/" + opt.scene + "/" + "test_real",
                                 "./datasets/" + opt.scene + "/" + "train_real"], mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_,
                                coord=False, depth=True, normal=False)
    elif opt.section == "test_sim_all":
        testset = CamLocDataset(["./datasets/" + opt.scene + "/" + "val_drone_sim",
                                 "./datasets/" + opt.scene + "/" + "val_sim",
                                 "./datasets/" + opt.scene + "/" + "test_drone_sim"], mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_,
                                coord=False, depth=True, normal=False)
    elif opt.section == "sim_all":
        testset = CamLocDataset(["./datasets/" + opt.scene + "/" + "val_drone_sim",
                                 "./datasets/" + opt.scene + "/" + "val_sim",
                                 "./datasets/" + opt.scene + "/" + "test_drone_sim",
                                 "./datasets/" + opt.scene + "/" + "train_sim_aug"], mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_,
                                coord=False, depth=True, normal=False)
    else:
        testset = CamLocDataset("./datasets/" + opt.scene + "/" + opt.section, mode=min(opt.mode, 1),
                                sparse=sparse, augment=False, grayscale=grayscale_,
                                coord=False, depth=True, normal=False)

    testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=4)
    if 'unc' in network_in or 'midrep' in network_in:
        uncertainty = True
        if opt.save_map:
            map_path = "./datasets/" + opt.scene + "/" + opt.section
            map_path = os.path.join(map_path,
                                    'depth_uncertainty_map_' + os.path.basename(network_in).replace('.net', '').replace(
                                        '.pth', ''))
            if not os.path.exists(map_path):
                os.makedirs(map_path)
    else:
        uncertainty = False
        if opt.save_map:
            map_path = "./datasets/" + opt.scene + "/" + opt.section
            map_path = os.path.join(map_path,
                                    'depth_prediction_map_' + os.path.basename(network_in).replace('.net', '').replace('.pth',
                                                                                                                 ''))
            if not os.path.exists(map_path):
                os.makedirs(map_path)
    if 'comballaz' in opt.scene.lower() or 'epfl' in opt.scene.lower():
        network = TransPoseNet(torch.zeros(1), opt.tiny, grayscale, num_task_channel=1,
                               num_pos_channel=1,
                               enc_add_res_block=2, dec_add_res_block=2)
    else:
        network = Network(torch.zeros((1)), opt.tiny)
    network.load_state_dict(torch.load(network_in))
    network = network.cuda()
    network.eval()

    test_log_path = os.path.join(os.path.dirname(network_in),
                                 'results_depth_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
    if os.path.exists(test_log_path):
        test_log = open(test_log_path, 'a')
        test_log.write('\n')
    else:
        test_log = open(test_log_path, 'w', 1)
    depth_log_path = os.path.join(os.path.dirname(network_in),
                                 'depths_%s_%s_%s.txt' % (opt.section, opt.scene, opt.session))
    if os.path.exists(depth_log_path):
        depth_log = open(depth_log_path, 'a')
        depth_log.write('\n')
    else:
        depth_log = open(depth_log_path, 'w', 1)

    print('Test images found: ', len(testset))

    # keep track of rotation and translation errors for calculation of the median error
    meanErrs = []
    medianErrs = []
    avg_time = 0

    pct20 = 0
    pct10 = 0
    pct5 = 0

    nodata_value = get_nodata_value(opt.scene)
    pixel_grid = get_pixel_grid(network.OUTPUT_SUBSAMPLE)

    nodata_value = get_nodata_value(opt.scene)

    with torch.no_grad():

        for image, gt_pose, init, focal_length, file in testset_loader:

            focal_length = float(focal_length[0])
            file = file[0].split('/')[-1]  # remove path from file name
            gt_pose = gt_pose[0]
            gt_depth = init.squeeze(1)  # [B, H_ds, W_ds]
            image = image.cuda()
            start_time = time.time()

            # predict scene coordinates and neural guidance
            if 'supercon' in network_in:
                depth_map, _ = network(image, True)
            else:
                depth_map = network(image)

            # scene_coordinates = init  # DEBUG
            if uncertainty:
                depth_map_ = depth_map  # [B, 2, H, W]
                depth_map = depth_map_[:, 0, :, :]  # [B, H, W]

            # check for invalid/unknown ground truth scene coordinates
            mask_gt_coords_valdata = pick_valid_points(gt_depth.view(gt_depth.size(0), -1), nodata_value,
                                                       boolean=True)  # [B, H_ds*W_ds]
            mask_gt_coords_nodata = torch.logical_not(mask_gt_coords_valdata)  # [B, H_ds*W_ds]

            mask_gt_coords_valdata = mask_gt_coords_valdata.view(gt_depth.shape)  # [B, H_ds*W_ds]
            mask_gt_coords_nodata = mask_gt_coords_nodata.view(gt_depth.shape)  # [B, H_ds*W_ds]

            regress_error = torch.abs(gt_depth - depth_map.cpu())  # [B, H_ds, W_ds]

            if opt.save_map:
                if uncertainty:
                    prediction_map = depth_map_[:, -1, :, :].cpu()  # [1, H, W]
                    prediction_map = torch.cat([prediction_map, regress_error], dim=0)  # [2, H_ds, W_ds]
                else:
                    prediction_map = regress_error

            avg_time += time.time() - start_time

            # calculate depth errors
            assert mask_gt_coords_valdata.size(0) == 1  # batch-size is 1. For ease of data processing
            regress_error_valid = regress_error[mask_gt_coords_valdata]  # [X]
            regress_error_valid_np = regress_error.detach().numpy()

            median_err = np.median(regress_error_valid_np)
            mean_err = np.mean(regress_error_valid_np)

            print("\nDepth error: median {:.1f}m, mean {:.1f}m, min {:.1f}m, max {:.1f}m".format(
                np.median(regress_error_valid_np), np.mean(regress_error_valid_np),
                np.min(regress_error_valid_np), np.max(regress_error_valid_np)))

            meanErrs.append(mean_err)
            medianErrs.append(median_err)

            if opt.save_map:
                output_info = list([prediction_map.cpu().numpy(), np.array([median_err, mean_err])])
                np.savez_compressed(os.path.join(map_path, file.replace('.png', '')), a=output_info[0],
                                    b=output_info[1])

            if median_err < 20:
                pct20 += 1
            if median_err < 10:
                pct10 += 1
            if median_err < 5:
                pct5 += 1

            depth_log.write("%s %f %f\n" % (
                file, median_err, mean_err))

    median_idx = int(len(meanErrs) / 2)
    medianErrs.sort()
    meanErrs.sort()
    avg_time /= len(meanErrs)

    print("\n===================================================")
    print("\nTest complete.")

    print('\nAccuracy:')
    print('\n20m: %.1f%%' % (pct20 / len(meanErrs) * 100))
    print('10m: %.1f%%' % (pct10 / len(meanErrs) * 100))
    print('5m: %.1f%%' % (pct5 / len(meanErrs) * 100))

    print("\nMean Error: %.1fm, Median Error: %.1fm" % (meanErrs[median_idx], medianErrs[median_idx]))
    print("Avg. processing time: %4.1fms" % (avg_time * 1000))
    test_log.write('%f %f %f\n' % (meanErrs[median_idx], medianErrs[median_idx], avg_time))

    test_log.close()
    depth_log.close()

    all_results_log.write('Separated test log: {:s} \n'.format(test_log_path))
    all_results_log.write('%f %f %f\n' % (meanErrs[median_idx], medianErrs[median_idx], avg_time))
all_results_log.close()
