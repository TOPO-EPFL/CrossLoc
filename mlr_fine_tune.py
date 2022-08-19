import pdb

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR

import time
import argparse
import os
import shutil
import sys
import git
import logging

PROJECT_DIR = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, PROJECT_DIR)
from dataloader.dataloader import CamLocDataset
from dataloader.dataloader_supercon import CamLocDatasetSupercon
from networks.networks import Network, TransPoseNet, ProjHead
from utils import get_nodata_value, get_pixel_grid, set_random_seed
from utils import get_cam_mat, get_label_mean, scene_coords_regression_loss, depth_regression_loss, normal_regression_loss
from utils import get_supercon_dataloader, read_training_log
from supercon_loss import SuperConLoss


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

    parser.add_argument('--drone_match', action='store_true',
                        help='to use the drone trajectory data, including synthetic and real matches')

    parser.add_argument('--real_only', action='store_true',
                        help='to use real data only, contrastive learning is forced to be turned off')

    parser.add_argument('--real_chunk', type=float, default=1.0,
                        help='to chunk the real data with given proportion')

    parser.add_argument('--task', type=str, required=True,
                        help='specify the single regression task, should be "coord", "depth" or "normal"')

    # Network structure
    parser.add_argument('--network_in', type=str, default=None,
                        help='file name of a network initialized for the scene')

    parser.add_argument('--tiny', '-tiny', action='store_true',
                        help='train a model with massively reduced capacity for a low memory footprint.')

    parser.add_argument('--depth_backbone', type=str, default=None,
                        help='to use the pretrained depth backbone model')

    parser.add_argument('--normal_backbone', type=str, default=None,
                        help='to use the pretrained surface normal backbone model')

    # Optimizer
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='number of training iterations, i.e. number of model updates')

    parser.add_argument('--learningrate', '-lr', type=float, default=0.00005,
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
                        help='coord only, switch to reprojection error optimization when predicted scene coordinate is within this tolerance threshold to the ground truth scene coordinate, in meters')

    parser.add_argument('--mindepth', '-mind', type=float, default=0.1,
                        help='coord: enforce predicted scene coordinates to be this far in front of the camera plane, in meters,'
                             'depth: enlarge loss weight for pixels violating the minimum depth constraint, in meters')

    parser.add_argument('--softclamp', '-sc', type=float, default=100,
                        help='coord: robust square root loss after this threshold, applied to reprojection error in pixels,'
                             'depth: robust square root loss after this threshold, in meters,'
                             'normal: robust square root loss for spherical loss if the angle error is beyond the threshold, in degrees')

    parser.add_argument('--hardclamp', '-hc', type=float, default=1000,
                        help='coord: clamp loss with this threshold, applied to reprojection error in pixels,'
                             'depth: the maximum depth regression error for valid/good predictions, in meters,'
                             'normal: the maximum surface normal regression angle error for valid/good predictions, in degrees')

    """Uncertainty loss parameter"""
    parser.add_argument('--uncertainty', '-uncertainty', action='store_true',
                        help='enable uncertainty learning')

    """Custom parameters for this training"""
    parser.add_argument('--sampling_pos_cross_dom', type=int, default=2,
                        help='choose at most this number of cross domain positive samples')

    parser.add_argument('--sampling_pos_in_dom', type=int, default=2,
                        help='choose at most this number of in domain positive samples')

    parser.add_argument('--sampling_neg_cross_dom', type=int, default=2,
                        help='choose at most this number of cross domain negative samples')

    parser.add_argument('--sampling_neg_in_dom', type=int, default=2,
                        help='choose at most this number of in domain negative samples')

    parser.add_argument('--supercon_weight', type=float, required=True,
                        help='weight for contrastive learning loss')

    parser.add_argument('--supercon_temperature', type=float, default=0.07,
                        help='temperature for contrastive learning loss')

    opt = parser.parse_args()

    return opt


def _config_directory(opt):
    """
    Configure directory to save model (task specific).
    """
    basename = opt.scene + '-{:s}'.format(opt.task) + '-mlr'
    if opt.session != '':
        basename += '-s' + opt.session
    if opt.grayscale:
        basename += '-gray'
    if opt.uncertainty:
        basename += '-unc'
    if opt.learningrate >= 1e-4:
        basename += '-e{:d}-lr{:.4f}'.format(opt.epochs, opt.learningrate)
    else:
        basename += '-e{:d}-lr{:.6f}'.format(opt.epochs, opt.learningrate)
    if opt.synthetic_only:
        basename += '-sim_only'
    elif opt.real_only:
        basename += '-real_only'
    else:
        if opt.drone_match:
            basename += '-drone_match'
        if opt.supercon_weight > 0:
            basename += '-supercon-w{:.2f}-t{:.2f}-pc{:d}-pi{:d}-nc{:d}-ni{:d}'.format(opt.supercon_weight,
                                                                                       opt.supercon_temperature,
                                                                                       opt.sampling_pos_cross_dom,
                                                                                       opt.sampling_pos_in_dom,
                                                                                       opt.sampling_neg_cross_dom,
                                                                                       opt.sampling_neg_in_dom)
        else:
            basename += '-vanilla'
    if opt.real_chunk < 1.0 and not opt.synthetic_only:
        basename += '-rc{:.2f}'.format(opt.real_chunk)
    if opt.translated:
        basename += '-translated'
    if opt.tiny:
        basename += '-tiny'

    # load pre-trained backbone models
    mlr_network_ls = []
    str_mlr = ''
    if opt.depth_backbone is not None:
        assert os.path.isfile(opt.depth_backbone)
        str_mlr += '_depth'
        mlr_network_ls.append(os.path.abspath(opt.depth_backbone))
    if opt.normal_backbone is not None:
        assert os.path.isfile(opt.normal_backbone)
        str_mlr += '_normal'
        mlr_network_ls.append(os.path.abspath(opt.normal_backbone))
    if len(str_mlr):
        basename += '-mlr' + str_mlr

    if opt.network_in is not None:
        basename += '-resume'

    # now = datetime.now()
    # start_time = now.strftime("-T%H.%M.%S-%d.%m.%y")
    # basename += start_time

    output_dir = os.path.abspath(os.path.join(PROJECT_DIR, 'output', basename))
    ckpt_output_dir = os.path.abspath(os.path.join(opt.ckpt_dir, basename)) if len(opt.ckpt_dir) else output_dir

    # try if auto_resume works
    if opt.auto_resume:
        flag_0 = os.path.exists(output_dir)
        flag_1 = os.path.exists(os.path.join(output_dir, 'output.log'))
        flag_2 = os.path.exists(os.path.abspath(os.path.join(output_dir, 'model.net')))
        if flag_0 and flag_1 and flag_2:
            pass
        else:
            opt.auto_resume = False
            print("Auto resume does not work. The script degenerates to normal training startup.")

    if opt.auto_resume:
        # Automatic resume training
        assert os.path.exists(output_dir), "output_dir at {:s} is not found! Cannot resume training.".format(output_dir)

        assert os.path.exists(os.path.join(output_dir, 'output.log'))

        existing_model_path = os.path.abspath(os.path.join(output_dir, 'model.net'))
        assert os.path.exists(existing_model_path), "Expected model weight at {:s} is not found!".format(
            existing_model_path)
        opt.network_in = existing_model_path

        os.makedirs(ckpt_output_dir, exist_ok=True)

    else:
        # Otherwise (usually start a new training from scratch)
        if os.path.exists(output_dir):
            key = input('Output directory already exists! Overwrite? (y/n)')
            if key.lower() == 'y':
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        if os.path.exists(ckpt_output_dir):
            shutil.rmtree(ckpt_output_dir)
            os.makedirs(ckpt_output_dir)
        else:
            os.makedirs(ckpt_output_dir)

    return output_dir, ckpt_output_dir, mlr_network_ls


def _config_log(opt):
    """
    Set configurations about logging to keep track of training progress.
    """
    output_dir, ckpt_output_dir, mlr_network_ls = _config_directory(opt)

    log_file = os.path.join(output_dir, 'output.log')
    if opt.auto_resume:
        file_handler = logging.FileHandler(log_file, mode='a')
    else:
        file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    if opt.auto_resume:
        logging.info('***** Automatic resume training from {:s} *****'.format(output_dir))
    else:
        logging.info('***** A new training has been started *****')
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: %s' % repo.head.object.hexsha)
    logging.info('Path to save data: {:s}'.format(output_dir))
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    logging.info('Arg parser: ')
    logging.info(opt)
    logging.info('Saving model to {:s}'.format(output_dir))
    logging.info('Saving checkpoint model to {:s}'.format(ckpt_output_dir))

    return output_dir, ckpt_output_dir, mlr_network_ls


def _config_dataloader(scene, task, drone_match, translated, grayscale, synthetic_only, real_only,
                       real_chunk, supercon_weight,
                       sampling_pos_cross_dom, sampling_pos_in_dom, sampling_neg_cross_dom, sampling_neg_in_dom,
                       batch_size, nodata_value):
    """
    Configure dataloader (task specific).
    """
    if 'epfl' in scene.lower() or 'comballaz' in scene.lower():
        pass
    else:
        raise NotImplementedError

    # original dataset to calculate mean
    root_sim = "./datasets/" + scene + "/train_drone_sim" if drone_match else "./datasets/" + scene + "/train_sim_aug"
    root_real = "./datasets/" + scene + "/train_translated" if translated else "./datasets/" + scene + "/train_real"
    trainset_vanilla = CamLocDataset([root_sim, root_real], coord=True, depth=True, normal=True,
                                     augment=False, raw_image=False, mute=True, real_chunk=real_chunk)
    trainset_loader_vanilla = torch.utils.data.DataLoader(trainset_vanilla, shuffle=False, batch_size=1,
                                                          num_workers=mp.cpu_count() // 2, pin_memory=True,
                                                          collate_fn=trainset_vanilla.batch_resize)

    mean = get_label_mean(trainset_loader_vanilla, nodata_value, scene, task)
    flag_coord = task == 'coord'
    flag_depth = task == 'depth'
    flag_normal = task == 'normal'

    if synthetic_only or real_only:
        if synthetic_only:
            trainset = CamLocDataset(root_sim, coord=flag_coord, depth=flag_depth, normal=flag_normal,
                                     augment=True, grayscale=grayscale, raw_image=False)
            logging.info(
                "Warning: this training uses synthetic data only. {:d} iterations per epoch.".format(len(trainset)))
        else:
            trainset = CamLocDataset(root_real, coord=flag_coord, depth=flag_depth, normal=flag_normal,
                                     augment=True, grayscale=grayscale, raw_image=False, real_chunk=real_chunk)
            logging.info(
                "Warning: this training uses real data only. {:d} iterations per epoch.".format(len(trainset)))
        trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                      num_workers=mp.cpu_count() // 2,
                                                      pin_memory=True, collate_fn=trainset.batch_resize)

    else:
        if supercon_weight > 0:
            if drone_match:
                raise NotImplementedError
            trainset = CamLocDatasetSupercon(root_dir_sim=root_sim, root_dir_real=root_real,
                                             coord=flag_coord, depth=flag_depth, normal=flag_normal,
                                             augment=True, grayscale=grayscale, raw_image=False, real_chunk=real_chunk,
                                             supercon=True,
                                             sampling_pos_cross_dom_top_n=sampling_pos_cross_dom,
                                             sampling_pos_in_dom_top_n=sampling_pos_in_dom,
                                             sampling_neg_cross_dom_top_n=sampling_neg_cross_dom,
                                             sampling_neg_in_dom_top_n=sampling_neg_in_dom)
            trainset_loader = get_supercon_dataloader(trainset, shuffle=False)  # no need to re-shuffle for the first epoch
        else:
            trainset = CamLocDataset([root_sim, root_real], coord=flag_coord, depth=flag_depth, normal=flag_normal,
                                     augment=True, grayscale=grayscale, raw_image=False, real_chunk=real_chunk)
            trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                          num_workers=mp.cpu_count() // 2,
                                                          pin_memory=True, collate_fn=trainset.batch_resize)
            logging.info("This training uses vanilla mode (mixing sim & real data naively). {:d} iterations per epoch".format(
                len(trainset)))

    return trainset, trainset_loader, mean


def _config_network(scene, task, tiny, grayscale, uncertainty, learningrate, mean, auto_resume, network_in, output_dir,
                    mlr_network_ls):
    """
    Configure network and optimizer (task specific).
    """
    if 'comballaz' in scene.lower() or 'epfl' in scene.lower():
        if task == 'coord':
            num_task_channel = 3
        elif task == 'normal':
            num_task_channel = 2
        elif task == 'depth':
            num_task_channel = 1
        else:
            raise NotImplementedError
        if uncertainty:
            num_pos_channel = 1
        else:
            num_pos_channel = 0
        network = TransPoseNet(mean, tiny, grayscale, num_task_channel=num_task_channel,
                               num_pos_channel=num_pos_channel,
                               enc_add_res_block=2, dec_add_res_block=2, num_mlr=len(mlr_network_ls))
    else:
        network = Network(mean, tiny)
    if network_in is not None:
        if len(mlr_network_ls):
            network.load_state_dict(torch.load(network_in), strict=False)
            logging.info("Successfully loaded %s." % network_in)
            model_path = os.path.join(output_dir, 'model.net')
            # setup mlr modules
            for i, mlr_network_in in enumerate(mlr_network_ls):
                network.mlr_encoder_ls[i].load_state_dict(torch.load(mlr_network_in), strict=False)
                for param in network.mlr_encoder_ls[i].parameters():
                    param.requires_grad = False
                logging.info("Successfully loaded pretrained model %s." % mlr_network_in)
        else:
            network.load_state_dict(torch.load(network_in))
            logging.info("Successfully loaded %s." % network_in)
            if auto_resume:
                model_path = os.path.join(output_dir, 'model_auto_resume.net')
                torch.save(network.state_dict(), model_path)
            else:
                model_path = os.path.join(output_dir, 'model_resume.net')
    else:
        model_path = os.path.join(output_dir, 'model.net')
    network = network.cuda()
    network.train()

    optimizer = optim.Adam(network.parameters(), lr=learningrate)

    return network, optimizer, model_path


def main():
    """
    Main function.
    """

    """Initialization"""
    set_random_seed(2021)
    opt = _config_parser()
    output_dir, ckpt_output_dir, mlr_network_ls = _config_log(opt)

    nodata_value = get_nodata_value(opt.scene)

    trainset, trainset_loader, mean = _config_dataloader(opt.scene, opt.task, opt.drone_match, opt.translated, opt.grayscale,
                                                         opt.synthetic_only, opt.real_only, opt.real_chunk,
                                                         opt.supercon_weight,
                                                         opt.sampling_pos_cross_dom, opt.sampling_pos_in_dom,
                                                         opt.sampling_neg_cross_dom, opt.sampling_neg_in_dom,
                                                         opt.batch_size, nodata_value)

    network, optimizer, model_path = _config_network(opt.scene, opt.task, opt.tiny, opt.grayscale, opt.uncertainty,
                                                     opt.learningrate, mean,
                                                     opt.auto_resume, opt.network_in, output_dir, mlr_network_ls)
    if opt.supercon_weight:
        activation = {}

        def hook(module, activation_in, activation_out):
            activation['module_name'] = module.__class__
            if isinstance(activation_in, tuple):
                assert len(activation_in) == 1
                activation['activation_in'] = activation_in[0]
            elif isinstance(activation_in, torch.Tensor):
                activation['activation_in'] = activation_in
            else:
                raise NotImplementedError
            activation['activation_out'] = activation_out
            return None

        network.decoder_ls[0].register_forward_hook(hook)
        proj_head = ProjHead(in_channel=network.decoder_ls[0].in_channels, out_length=2048,
                             tiny=network.tiny).cuda()
        criterion_supercon = SuperConLoss(opt.supercon_temperature, force_normalization=True)

    # Learning rate could be problematic for the very small real dataset!
    if opt.no_lr_scheduling:
        # scheduler would be turned OFF
        scheduler = MultiStepLR(optimizer, [opt.epochs], gamma=1.0)
    else:
        scheduler = MultiStepLR(optimizer, [30, 60], gamma=0.5)

    pixel_grid = get_pixel_grid(network.OUTPUT_SUBSAMPLE)

    """Training loop"""
    epochs = opt.epochs
    if opt.auto_resume:
        iteration, start_epoch = read_training_log(os.path.join(output_dir, 'output.log'), len(trainset))
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
            if opt.uncertainty:
                predictions, uncertainty_map = torch.split(predictions, [network.num_task_channel, network.num_pos_channel], dim=1)  # [B, C, H, W] + [B, 1, H, W]
            else:
                uncertainty_map = None

            """Backward loop"""
            # regression loss
            reduction = None if opt.supercon_weight else 'mean'
            if opt.task == 'coord':
                loss, valid_pred_rate = scene_coords_regression_loss(opt.mindepth, opt.softclamp, opt.hardclamp,
                                                                     opt.inittolerance, opt.uncertainty,
                                                                     pixel_grid, nodata_value, cam_mat,
                                                                     predictions, uncertainty_map, gt_poses, gt_labels,
                                                                     reduction)
            # elif opt.task == 'depth':
            #     loss, valid_pred_rate = depth_regression_loss(opt.mindepth, opt.softclamp, opt.hardclamp,
            #                                                   opt.uncertainty, nodata_value, predictions,
            #                                                   uncertainty_map, gt_labels, reduction)
            # elif opt.task == 'normal':
            #     loss, valid_pred_rate = normal_regression_loss(opt.softclamp, opt.hardclamp, opt.uncertainty,
            #                                                    nodata_value, predictions, uncertainty_map,
            #                                                    gt_labels, reduction)
            else:
                raise NotImplementedError

            # geometric supercon loss
            if opt.supercon_weight:
                mask_anchor, mask_positive, mask_negative, loss_weights = trainset.get_aux_info(idx)

                # info-NCE loss
                if 'debug' in opt.session.lower():
                    raise NotImplementedError
                else:
                    # Weight regression loss because of unbalanced dataset frequency
                    loss = (loss * loss_weights.cuda()).mean()
                    if len(loss_weights) > 1:
                        feats = proj_head(activation['activation_in'])
                        loss_supercon = criterion_supercon(feats[mask_anchor], feats[mask_positive], feats[mask_negative])
                        loss_supercon *= opt.supercon_weight
                        loss += loss_supercon
                    else:
                        loss_supercon = torch.zeros(1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            """Training process record."""
            batch_size = len(images)
            time_avg = (time.time() - start_time) / batch_size
            iteration = iteration + batch_size
            if opt.supercon_weight:
                logging.info(
                    'Iteration: %7d, Epoch: %3d, Total loss: %.2f, Supercon loss: %.2f, Valid: %.1f%%, Avg Time: %.3fs' % (
                        iteration, epoch, loss.item(), loss_supercon.item(), valid_pred_rate * 100, time_avg))
            else:
                logging.info(
                    'Iteration: %7d, Epoch: %3d, Total loss: %.2f, Valid: %.1f%%, Avg Time: %.3fs' % (
                        iteration, epoch, loss.item(), valid_pred_rate * 100, time_avg))

            if iteration > save_counter:
                logging.info('Saving snapshot of the network to %s.' % model_path)
                torch.save(network.state_dict(), model_path)
                save_counter = iteration + len(trainset)  # every one de-facto epoch
                epoch_de_facto += 1
                scheduler.step()

            # save checkpoint every 5 de-facto epochs
            if iteration > _last_ckpt_iteration + 5 * len(trainset) or _last_ckpt_iteration == 0:
                torch.save(network.state_dict(),
                           os.path.join(ckpt_output_dir, 'ckpt_iter_{:07d}.net'.format(iteration)))
                _last_ckpt_iteration = iteration

        logging.info('Saving snapshot of the network to %s.' % model_path)
        torch.save(network.state_dict(), model_path)

        if opt.supercon_weight:
            trainset_loader = get_supercon_dataloader(trainset, shuffle=True)

    logging.info('Done without errors.')
    torch.save(None, os.path.join(output_dir, 'FLAG_training_done.nodata'))
    torch.save(None, os.path.join(ckpt_output_dir, 'FLAG_training_done.nodata'))


if __name__ == '__main__':
    main()
