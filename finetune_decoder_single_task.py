import pdb

import torch


import time
import argparse
import os
import sys
import logging
from loss.coord import get_cam_mat, scene_coords_regression_loss
from loss.depth import depth_regression_loss
from loss.normal import normal_regression_loss
from loss.semantics import semantics_classification_loss, CrossEntropyLoss2d
from utils.learning import get_pixel_grid, get_nodata_value, set_random_seed, config_dataloader, config_network
from utils.io import read_training_log, config_log, check_encoders

PROJECT_DIR = os.path.abspath(os.path.join(__file__, '..'))
# sys.path.insert(0, PROJECT_DIR)


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
                        help='batch size of the dataloader.')

    parser.add_argument('--grayscale', '-grayscale', action='store_true',
                        help='use grayscale image as model input')

    parser.add_argument('--real_data_domain', type=str, default='in_place',
                        help="to select the domain of pairwise sim-to-real data, i.e., in_place or out_of_place")

    parser.add_argument('--real_data_chunk', type=float, default=1.0,
                        help='to chunk the pairwise sim-to-real data with given proportion')

    parser.add_argument('--real_only', action='store_true',
                        help='to use real data only')

    parser.add_argument('--sim_data_chunk', type=float, default=0.0,
                        help='to chunk the task-agnostic LHS synthetic data with given proportion')

    parser.add_argument('--task', type=str, required=True,
                        help='specify the single regression task, should be "coord", "depth", "normal" or "semantics"')

    parser.add_argument('--epoch_plus', '-epoch_plus', action='store_true',
                        help='extend training by epochs, a already well-trained model w/ the same configurations must '
                             'be found (except for the epochs).')

    parser.add_argument('--encoders', type=str, nargs='+', required=True,
                        help='pretrained encoders to use. note that if the coordinate regression encoder is found by'
                             'the folder name, its weight is automatically loaded for the new network initialization.'
                             'this argument is effectively only for the first time the training is started.'
                             'acceptable: coord, depth, normal, semantics')

    parser.add_argument('--coord_weight', required=True,
                        help='path to pretrained MLR encoder')

    parser.add_argument('--depth_weight', required=True,
                        help='path to pretrained MLR encoder')

    parser.add_argument('--normal_weight', required=True,
                        help='path to pretrained MLR encoder')

    parser.add_argument('--semantics_weight', required=True,
                        help='path to pretrained MLR encoder')

    parser.add_argument('--reuse_coord_encoder', default=False, action="store_true",
                        help="to re-use coordinate encoder (frozen) during training")

    parser.add_argument('--unfreeze_coord_encoder', default=False, action="store_true",
                        help="to re-use coordinate encoder (trainable) during training")

    # Network structure
    parser.add_argument('--network_in', type=str, default=None,
                        help='file name of a network initialized for the scene')

    parser.add_argument('--tiny', '-tiny', action='store_true',
                        help='train a model with massively reduced capacity for a low memory footprint.')

    parser.add_argument('--fullsize', '-fullsize', action='store_true',
                        help='to output fillsize prediction w/o down-sampling.')

    # Optimizer
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='number of training iterations, i.e. number of model updates')

    parser.add_argument('--learningrate', '-lr', type=float, default=0.0002,
                        help='learning rate')

    parser.add_argument('--no_lr_scheduling', action='store_true',
                        help='To disable learning rate scheduler.')

    """I/O parameters"""
    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, useful to separate different runs of a script')

    parser.add_argument('--ckpt_dir', type=str, default='',
                        help="directory to save checkpoint models.")

    parser.add_argument('--auto_resume', action='store_true',
                        help='resume training, including: to load the latest weight and keep the checkpoint directory, '
                             'to read and concatenate output logging and tune the scheduler accordingly')

    """Scene coordinate regression task parameters (taken from DSAC*)"""
    # Note: in depth training mode, mindepth, softclamp and hardclamp parameters are used.
    # in normal training mode, softclamp and hardclamp parameters are used.
    parser.add_argument('--inittolerance', '-itol', type=float, default=50.0,
                        help='coord only, turn on reprojection error optimization when the predicted scene coordinates'
                             'projected into camera coord frame are within this tolerance, in meters')

    parser.add_argument('--mindepth', '-mind', type=float, default=0.1,
                        help='coord: enforce predicted scene coordinates projected into camera coord frame '
                             '       to be this far in front of the camera plane, in meters;'
                             'depth: min depth threshold for valid prediction, in meters')

    parser.add_argument('--softclamp', '-sc', type=float, default=100,
                        help='coord only, robust square root loss after this threshold, applied to '
                             'reprojection error, in pixels.')

    parser.add_argument('--hardclamp', '-hc', type=float, default=1000,
                        help='coord: clamp loss with this threshold, applied to reprojection error, in pixels;'
                             'depth: max error threshold for valid prediction (not for loss), in meters;'
                             'normal: max error threshold for valid prediction (not for loss), in degrees')

    parser.add_argument('--debug', action='store_true',
                        help='enter debug mode')

    """Uncertainty loss parameter"""
    parser.add_argument('--uncertainty', '-uncertainty', default=None, type=str,
                        help='enable uncertainty learning')

    opt = parser.parse_args()

    if isinstance(opt.uncertainty, str):
        if opt.uncertainty.lower() == 'none':
            opt.uncertainty = None
        elif opt.uncertainty.lower() == 'mle':
            opt.uncertainty = 'MLE'

    assert opt.uncertainty in [None, 'MLE'], \
        '--uncertainty {} is not supported!'.format(opt.uncertainty)

    assert opt.real_data_domain in ['in_place', 'out_of_place'], \
        '--real_data_domain {:} is not supported!'.format(opt.real_data_domain)

    return opt


def get_output_path(opt):
    """
    Task-specific output directory name.
    """
    basename = opt.scene + '-{:s}'.format(opt.task)
    basename += '-decoder_' + '_'.join(opt.encoders)
    if opt.reuse_coord_encoder:
        if opt.unfreeze_coord_encoder:
            basename = basename.replace('_coord_', '_coord_free_')
        else:
            basename = basename.replace('_coord_', '_coord_frozen_')
    else:
        basename = basename.replace('_coord_', '_')
    if opt.session != '':
        basename += '-s' + opt.session
    if opt.grayscale:
        basename += '-gray'
    if opt.uncertainty is None:
        basename += '-no_unc'
    else:
        basename += '-unc-{:s}'.format(opt.uncertainty)
    if opt.fullsize:
        basename += '-fullsize'
    if opt.learningrate >= 1e-4:
        basename += '-e{:d}-lr{:.4f}'.format(opt.epochs, opt.learningrate)
    else:
        basename += '-e{:d}-lr{:.6f}'.format(opt.epochs, opt.learningrate)
    if opt.real_data_chunk == 0.0:
        assert opt.sim_data_chunk > 0
        if '-ft0.00' in opt.session:
            # encoders are not fine-tuned w/ any real data
            basename += '-zero_shot'
        else:
            basename += '-sim_only'
        basename += '-sc{:.2f}'.format(opt.sim_data_chunk)
    else:
        assert opt.sim_data_chunk == 0.0  # disable LHS sim data when pairwise sim-to-real data is used.
        if opt.real_only:
            basename += '-real_only'
        else:
            basename += '-pairwise'
        if opt.real_data_domain == 'in_place':
            basename += '-ip'
        elif opt.real_data_domain == 'out_of_place':
            basename += '-oop'
        else:
            raise NotImplementedError
        basename += '-rc{:.2f}'.format(opt.real_data_chunk)
    if opt.tiny:
        basename += '-tiny'
    if opt.network_in is not None:
        basename += '-resume'
    if opt.debug:
        basename += '-DEBUG'

    # now = datetime.now()
    # start_time = now.strftime("-T%H.%M.%S-%d.%m.%y")
    # basename += start_time

    output_dir = os.path.abspath(os.path.join(PROJECT_DIR, 'output', basename))

    return output_dir


def main():
    """
    Main function.
    """

    """Initialization"""
    set_random_seed(2021)
    opt = _config_parser()
    output_dir, ckpt_output_dir = config_log(opt, get_output_path(opt))

    if opt.network_in is not None:
        # network_in could be superimposed by forceful network loading, auto resume and epoch plus options
        logging.info("opt.network_in at {:} has been properly specified".format(opt.network_in))
        logging.info("Encoder list {} is ineffective!".format(opt.encoders))
    else:
        logging.info("Encoder list {} is effective!".format(opt.encoders))
    encoders_path = check_encoders(opt.encoders, opt.coord_weight, opt.depth_weight,
                                   opt.normal_weight, opt.semantics_weight)

    nodata_value = get_nodata_value(opt.scene)

    trainset, trainset_loader, mean = config_dataloader(opt.scene, opt.task, opt.grayscale,
                                                        opt.real_data_domain, opt.real_data_chunk, opt.sim_data_chunk,
                                                        opt.fullsize,
                                                        opt.batch_size, nodata_value, opt.real_only)

    network, optimizer, model_path, scheduler = config_network(opt.scene, opt.task, opt.tiny, opt.grayscale,
                                                               opt.uncertainty, opt.fullsize, mean,
                                                               opt.learningrate, opt.no_lr_scheduling,
                                                               opt.auto_resume, opt.epoch_plus,
                                                               opt.network_in, output_dir,
                                                               encoders_path,
                                                               opt.reuse_coord_encoder, opt.unfreeze_coord_encoder)

    save_period = 5  # to save a checkpoint model every N epochs
    if opt.task == 'coord':
        pixel_grid = get_pixel_grid(network.OUTPUT_SUBSAMPLE)
    elif opt.task == 'semantics':
        semantic_criterion = CrossEntropyLoss2d()
        save_period = 1

    """Training loop"""
    epochs = opt.epochs
    if opt.auto_resume or opt.epoch_plus:
        iteration, start_epoch = read_training_log(os.path.join(os.path.dirname(opt.network_in), 'output.log'),
                                                   len(trainset))
        save_counter = (start_epoch + 1) * len(trainset)
        epoch_de_facto = start_epoch
        _last_ckpt_iteration = (start_epoch // 5 * 5) * len(trainset)

        # refresh learning rate
        optimizer.step()
        optimizer.zero_grad()
        [scheduler.step() for e in range(start_epoch)]
    else:
        iteration, start_epoch, save_counter, epoch_de_facto, _last_ckpt_iteration = 0, 0, 0, 0, 0

    for epoch in range(epochs):

        if epoch < start_epoch:
            continue
        else:
            logging.info("Optimizer works effectively with a learning rate of {:.6f}".format(
                optimizer.param_groups[0]['lr']))

        logging.info("=== Epoch: %d ======================================" % epoch)

        for idx, (images, gt_poses, gt_labels, focal_lengths, file_names) in enumerate(trainset_loader):
            start_time = time.time()

            """Data pre-processing"""
            focal_length = float(focal_lengths.view(-1)[0])
            """
            @images         [B, C, H, W] ---> [B, 3, 480, 720] by default w/o augmentation, RGB image
            @gt_poses       [B, 4, 4], camera to world matrix
            @gt_labels      [B, C, H_ds, W_ds] ---> [B, C, 60, 90] by default w/o augmentation
            @focal_length   [1], adapted to augmentation
            @file_names     a list size of B
            """
            cam_mat = get_cam_mat(images.size(3), images.size(2), focal_length)
            gt_poses = gt_poses.cuda()
            gt_labels = gt_labels.cuda()

            """Forward pass"""
            predictions = network(images.cuda())
            if opt.fullsize:
                assert predictions.size(2) == images.size(2) and predictions.size(3) == images.size(3)
                assert predictions.size(2) == gt_labels.size(2) and predictions.size(3) == gt_labels.size(3)
            if opt.uncertainty is None:
                uncertainty_map = None
            elif opt.uncertainty == 'MLE':
                predictions, uncertainty_map = torch.split(predictions,
                                                           [network.num_task_channel, network.num_pos_channel],
                                                           dim=1)  # typically [B, C, H, W] + [B, 1, H, W]
            else:
                raise NotImplementedError

            """Backward loop"""
            # regression loss
            reduction = 'mean'
            if opt.task == 'coord':
                loss, valid_pred_rate = scene_coords_regression_loss(opt.mindepth, opt.softclamp, opt.hardclamp,
                                                                     opt.inittolerance, opt.uncertainty,
                                                                     pixel_grid, nodata_value, cam_mat,
                                                                     predictions, uncertainty_map, gt_poses, gt_labels,
                                                                     reduction)
            elif opt.task == 'depth':
                loss, valid_pred_rate = depth_regression_loss(opt.mindepth, opt.hardclamp,
                                                              opt.uncertainty, nodata_value, predictions,
                                                              uncertainty_map, gt_labels, reduction)
            elif opt.task == 'normal':
                loss, valid_pred_rate = normal_regression_loss(opt.hardclamp, opt.uncertainty,
                                                               nodata_value, predictions, uncertainty_map,
                                                               gt_labels, reduction)
            elif opt.task == 'semantics':
                loss, valid_pred_rate = semantics_classification_loss(opt.uncertainty, predictions, uncertainty_map,
                                                                      gt_labels, semantic_criterion, reduction)
            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            """Training process record."""
            batch_size = len(images)
            time_avg = (time.time() - start_time) / batch_size
            iteration = iteration + batch_size
            logging.info(
                'Iteration: %7d, Epoch: %3d, Total loss: %.2f, Valid: %.1f%%, Avg Time: %.3fs' % (
                    iteration, epoch, loss.item(), valid_pred_rate * 100, time_avg))

            if iteration > save_counter:
                logging.info('Saving snapshot of the network to %s.' % model_path)
                torch.save(network.state_dict(), model_path)
                save_counter = iteration + len(trainset)  # every one de-facto epoch
                epoch_de_facto += 1
                scheduler.step()

            # save checkpoint every N de-facto epochs
            if iteration > _last_ckpt_iteration + save_period * len(trainset) or _last_ckpt_iteration == 0:
                torch.save(network.state_dict(),
                           os.path.join(ckpt_output_dir, 'ckpt_iter_{:07d}.net'.format(iteration)))
                _last_ckpt_iteration = iteration

        logging.info('Saving snapshot of the network to %s.' % model_path)
        torch.save(network.state_dict(), model_path)

    logging.info('Done without errors.')
    torch.save(None, os.path.join(output_dir, 'FLAG_training_done.nodata'))
    torch.save(None, os.path.join(ckpt_output_dir, 'FLAG_training_done.nodata'))


if __name__ == '__main__':
    main()
