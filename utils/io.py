import glob
import logging
import os
import copy
import pdb
import re
import shutil
import sys

import git
import numpy as np


def safe_printout(words):
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

    if len(iterations_ls) and len(epochs_ls):
        last_iteration = np.max(iterations_ls)
        last_epoch = np.max(epochs_ls)

        # sanity check for the read statistics
        assert abs(last_iteration // iter_per_epoch - last_epoch) <= 5, \
            "Last iteration {:d} does not match last epoch {:d} with iteration per epoch being {:d}.".format(
                last_iteration, last_epoch, iter_per_epoch)
    else:
        safe_printout("Maybe this is an empty training log. Setting last_iteration and last_epoch to 0...")
        last_iteration = 0
        last_epoch = 0
    return last_iteration, last_epoch


def get_unique_file_name(file_path):
    """
    Get unique file name for unique mapping.
    The generated filename includes basename and section, e.g., XXX-LHS_00000_LHS.png@train_sim
    """
    file_section = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    unique_file_nm = os.path.basename(file_path) + '@' + file_section
    return unique_file_nm


def get_epoch_from_dirname(model_dirname):
    cur_epoch = re.findall(r'-e(?P<epoch>\d+)', model_dirname)
    if len(cur_epoch) == 1:
        cur_epoch = int(cur_epoch[0])
    else:
        cur_epoch = None
    return cur_epoch


def search_epoch_extension_model(output_dir):
    """
    Search folders to load the model weight for epoch extension experiments.
    """
    model_dirname = os.path.basename(output_dir)

    cur_epoch = get_epoch_from_dirname(model_dirname)
    key_str = '-e{:d}'.format(cur_epoch)
    model_dirname_prefix = model_dirname[:model_dirname.find(key_str)]
    model_dirname_suffix = model_dirname[model_dirname.find(key_str)+len(key_str):]

    all_stuff = glob.glob(os.path.abspath(os.path.join(output_dir, '../*')))  # list all neighboring file or folders
    allowed_folder_ls, found_epoch_ls = [], []
    for this_entry in all_stuff:
        if os.path.isdir(this_entry):
            if model_dirname_prefix in this_entry and model_dirname_suffix in this_entry:
                this_epoch = get_epoch_from_dirname(os.path.basename(this_entry))
                if this_epoch is not None:
                    flag_model = os.path.exists(os.path.join(this_entry, 'model.net')) or \
                                 os.path.exists(os.path.join(this_entry, 'model_epoch_plus_resume.net')) or \
                                 os.path.exists(os.path.join(this_entry, 'model_auto_resume.net')) or \
                                 os.path.exists(os.path.join(this_entry, 'model_resume.net'))
                    flag_done = os.path.exists(os.path.join(this_entry, 'FLAG_training_done.nodata'))
                    flag_log = os.path.exists(os.path.join(this_entry, 'output.log'))
                    if np.all([flag_model, flag_done, flag_log]):
                        allowed_folder_ls.append(this_entry)
                        found_epoch_ls.append(this_epoch)
    if len(allowed_folder_ls):
        print("Epoch extension: {:d} possible existing folders are found.".format(len(allowed_folder_ls)))
        idx = np.argmax(found_epoch_ls)
        select_model_dirname = allowed_folder_ls[idx]
        select_model_epoch = found_epoch_ls[idx]
        print("Checkpoint of epoch {:d} at {:s} will be loaded...".format(select_model_epoch,
                                                                                 select_model_dirname))
        return select_model_dirname
    else:
        raise Exception("ERROR! No plausible model to read for epoch extension experiments.")


def config_directory(output_dir, ckpt_dir, auto_resume, epoch_plus, default_network_in=None):
    """
    Configure directory to save model.
    """

    output_dir = os.path.abspath(output_dir)
    ckpt_output_dir = os.path.abspath(os.path.join(ckpt_dir, os.path.basename(output_dir))) if len(ckpt_dir) else output_dir

    # try if auto_resume works
    if auto_resume:
        flag_0 = os.path.exists(output_dir)
        flag_1 = os.path.exists(os.path.join(output_dir, 'output.log'))
        flag_2 = os.path.exists(os.path.abspath(os.path.join(output_dir, 'model.net'))) or os.path.exists(
            os.path.abspath(os.path.join(output_dir, 'model_epoch_plus_resume.net'))) or os.path.exists(
            os.path.abspath(os.path.join(output_dir, 'model_auto_resume.net'))) or os.path.exists(
            os.path.abspath(os.path.join(output_dir, 'model_resume.net')))
        if flag_0 and flag_1 and flag_2:
            pass
        else:
            auto_resume = False
    print("Effective auto resume: {}".format(auto_resume))

    # try if epoch_plus works
    # warning: epoch extension priority is lower than auto_resume
    # it is only effective when epoch extension is initiated for the first time
    # when no suitable checkpoint model is detected, an error is returned
    _epoch_plus = copy.copy(epoch_plus)
    if epoch_plus:
        if auto_resume:
            epoch_plus = False
        else:
            # resume from the most recent epoch
            epoch_plus_resume_dir = search_epoch_extension_model(output_dir)
    print("Effective epoch extension: {}".format(epoch_plus))

    if auto_resume or epoch_plus:
        if auto_resume:
            resume_dir = output_dir
        elif epoch_plus:
            resume_dir = epoch_plus_resume_dir
            os.makedirs(output_dir, exist_ok=True)
        # Automatic resume training
        assert os.path.exists(resume_dir), "resume_dir at {:s} is not found! Resume training failed.".format(resume_dir)

        assert os.path.exists(os.path.join(resume_dir, 'output.log'))

        # check if the last model is auto resumed training
        flag_last_auto_resume = os.path.exists(os.path.abspath(os.path.join(resume_dir, 'model_auto_resume.net')))

        if flag_last_auto_resume:
            # continue last auto resumed training weight
            existing_model_path = os.path.abspath(os.path.join(resume_dir, 'model_auto_resume.net'))
        else:
            if auto_resume:  # epoch_plus is False
                # continue with the most recent training weight
                if _epoch_plus:
                    existing_model_path = os.path.abspath(os.path.join(resume_dir, 'model_epoch_plus_resume.net'))
                else:
                    if default_network_in is None:
                        existing_model_path = os.path.abspath(os.path.join(resume_dir, 'model.net'))
                    else:
                        existing_model_path = os.path.abspath(os.path.join(resume_dir, 'model_resume.net'))
            else:  # auto_resume if False
                # continue using the most recent training weight
                if os.path.exists(os.path.abspath(os.path.join(resume_dir, 'model_epoch_plus_resume.net'))):
                    existing_model_path = os.path.abspath(os.path.join(resume_dir, 'model_epoch_plus_resume.net'))
                else:
                    if default_network_in is None:
                        existing_model_path = os.path.abspath(os.path.join(resume_dir, 'model.net'))
                    else:
                        existing_model_path = os.path.abspath(os.path.join(resume_dir, 'model_resume.net'))

        assert os.path.exists(existing_model_path), "Expected model weight at {:s} is not found!".format(
            existing_model_path)
        network_to_load = existing_model_path

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
        network_to_load = None

    return output_dir, ckpt_output_dir, network_to_load, auto_resume, epoch_plus


def config_log(opt, output_dirname):
    """
    Set configurations about logging to keep track of training progress.
    """
    output_dir, ckpt_output_dir, network_to_load, flag_auto_resume, flag_epoch_plus = config_directory(
        output_dirname, opt.ckpt_dir, opt.auto_resume, opt.epoch_plus, opt.network_in)

    # try to find a non-None network_in argument
    # when the network is initialized, we ensure it loads the specified network.
    if opt.network_in is not None and network_to_load is None:
        pass
    else:
        opt.network_in = network_to_load
    opt.auto_resume = flag_auto_resume
    opt.epoch_plus = flag_epoch_plus

    if opt.epoch_plus:
        shutil.copy2(os.path.join(os.path.dirname(network_to_load), 'output.log'),
                     os.path.join(output_dir, 'output.log'))
    log_file = os.path.join(output_dir, 'output.log')
    if opt.auto_resume or opt.epoch_plus:
        file_handler = logging.FileHandler(log_file, mode='a')
    else:
        file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    if opt.auto_resume:
        logging.info('***** Automatic resume training from {:s} *****'.format(opt.network_in))
    elif opt.epoch_plus:
        logging.info('***** Epoch extension resume training from {:s} *****'.format(opt.network_in))
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

    return output_dir, ckpt_output_dir


def check_encoders(encoders: list, coord_weight: str, depth_weight: str,
                   normal_weight: str, semantics_weight: str):
    """Check the models in the encoders list."""

    # sanity check
    for entry in encoders:
        assert entry in ['coord', 'depth', 'normal', 'semantics'], "encoder model {:s} is not supported!".format(entry)
    if 'coord' not in encoders:
        raise Exception("A coordinate regression network weight must be provided for decoder initialization!")
    encoders = np.unique(encoders).tolist()
    assert 1 <= len(encoders) <= 4

    # retrieve path
    encoders_path = []
    for entry in encoders:
        if entry == 'coord':
            assert os.path.exists(coord_weight)
            encoders_path.insert(0, coord_weight)  # the first one must be coord encoder
        elif entry == 'depth':
            assert os.path.exists(depth_weight)
            encoders_path.append(depth_weight)
        elif entry == 'normal':
            assert os.path.exists(normal_weight)
            encoders_path.append(normal_weight)
        elif entry == 'semantics':
            assert os.path.exists(semantics_weight)
            encoders_path.append(semantics_weight)
    logging.info("{:d} network weights are to be loaded for reuse".format(len(encoders_path)))
    return encoders_path
