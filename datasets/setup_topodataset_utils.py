import glob
import os
import pdb

import pyproj
import shutil
import itertools
import torch
import open3d as o3d
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

focallength = 480.0  # focal length in pixel
stride = 8  # downsampling rate

# image resolution
width = 720
height = 480


def get_rotation_ned_in_ecef(lon, lat):
    """
    @param: lon, lat Longitude and latitude in degree
    @return: 3x3 rotation matrix of heading-pith-roll NED in ECEF coordinate system
    Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
    Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
    """
    # describe NED in ECEF
    lon = lon * np.pi / 180.0
    lat = lat * np.pi / 180.0
    # manual computation
    R_N0 = np.array([[np.cos(lon), -np.sin(lon), 0],
                     [np.sin(lon), np.cos(lon), 0],
                     [0, 0, 1]])
    R__E1 = np.array([[np.cos(-lat - np.pi / 2), 0, np.sin(-lat - np.pi / 2)],
                      [0, 1, 0],
                      [-np.sin(-lat - np.pi / 2), 0, np.cos(-lat - np.pi / 2)]])
    NED = np.matmul(R_N0, R__E1)
    assert abs(np.linalg.det(
        NED) - 1.0) < 1e-6, 'NED in NCEF rotation mat. does not have unit determinant, it is: {:.2f}'.format(
        np.linalg.det(NED))
    return NED


def ecef_to_geographic(x, y, z):
    # Careful: here we need to use lat,lon
    lat, lon, alt = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979").transform(x, y, z)
    return [lon, lat, alt]


def get_pose_mat(cesium_pose):
    """
    Get 4x4 homogeneous matrix from Cesium-defined pose
    @input: cesium_pose 6d ndarray, [lat, lon, h, yaw, pitch, roll]
    lat, lon, h are in ECEF coordinate system
    yaw, pitch, roll are in degress
    @output: 4x4 homogeneous extrinsic camera matrix
    """
    x, y, z, yaw, pitch, roll = cesium_pose  # no need to do local conversion when in ECEF
    lon, lat, alt = ecef_to_geographic(x, y, z)
    rot_ned_in_ecef = get_rotation_ned_in_ecef(lon, lat)
    rot_pose_in_ned = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
    r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    # transform coordiante system from NED to standard camera sys.
    r = r[0:3, [1, 2, 0]]
    r = np.concatenate((r, np.array([[x, y, z]]).transpose()), axis=1)
    r = np.concatenate((r, np.array([[0, 0, 0, 1]])), axis=0)
    return r


def mkdir(directory):
    """Checks whether the directory exists and creates it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def _mp_func_coor(dat_src_path, dat_path, origin, mp_counter, mp_lock, ttl_size):
    """Multiprocessing backbone function for coordinate extraction"""
    pc = np.load(dat_src_path)  # [H, W, 3] = [480, 720, 3]
    assert pc.shape[0] == height and pc.shape[1] == width
    pc = pc[range(0, height, stride), :, :]
    pc = pc[:, range(0, width, stride), :].transpose((2, 0, 1))  # [3, H_ds, W_ds]
    pc = (pc - origin[:, None, None]) * (pc != -1) + pc * (pc == -1)  # [3, :, :]
    pc = torch.tensor(pc, dtype=torch.float)  # [3, H_ds, W_ds] = [3, 60, 90]
    torch.save(pc, dat_path)
    with mp_lock:
        mp_counter.value += 1
        print("\rCoordinate extraction progress: {:d}/{:d}".format(mp_counter.value, ttl_size), end="", flush=True)


def _mp_func_depth(init_src_path, depth_path, pose_txt_path, mp_counter, mp_lock, ttl_size):
    """Multiprocessing backbone function for z-buffer depth extraction from point clouds and camera extrinsics"""
    pc = torch.load(init_src_path)  # [3, H_ds, W_ds] = [3, 60, 90]
    cam_to_world = np.loadtxt(pose_txt_path)  # [4, 4]
    world_to_cam = np.linalg.inv(cam_to_world)  # [4, 4]

    world_coords = pc.reshape(3, -1)  # [3, N]
    ones = torch.ones((1, world_coords.size(1)))  # [1, N]
    world_coords = torch.cat([world_coords, ones], dim=0)  # [4, N]

    depth_z_buffer = torch.matmul(torch.tensor(world_to_cam[2:3, :]).float(), world_coords.float())  # [1, N] <- [1, 4] x [4, N]
    depth_z_buffer = depth_z_buffer.view(pc.size(1), pc.size(2))  # [H_ds, W_ds] = [60, 90]
    depth_z_buffer = depth_z_buffer * (pc[0] != -1) + pc[0] * (pc[0] == -1)  # [H_ds, W_ds] = [60, 90]
    torch.save(depth_z_buffer, depth_path)
    with mp_lock:
        mp_counter.value += 1
        print("\rDepth extraction progress: {:d}/{:d}".format(mp_counter.value, ttl_size), end="", flush=True)


def _mp_func_normal(init_src_path, normal_path, pose_txt_path, mp_counter, mp_lock, ttl_size):
    """Multiprocessing backbone function for z-buffer depth extraction from point clouds and camera extrinsics"""
    pc = torch.load(init_src_path)  # [3, H_ds, W_ds] = [3, 60, 90]
    cam_to_world = np.loadtxt(pose_txt_path)  # [4, 4]
    world_to_cam = np.linalg.inv(cam_to_world)  # [4, 4]

    world_coords = pc.reshape(3, -1)  # [3, N]
    ones = torch.ones((1, world_coords.size(1)))  # [1, N]
    world_coords = torch.cat([world_coords, ones], dim=0)  # [4, N]

    mask_has_data = np.sum(pc.numpy().reshape(3, -1).transpose(1, 0), axis=1) != -3  # [N]
    cam_coords = torch.matmul(torch.tensor(world_to_cam[:3, :]).float(), world_coords.float())  # [3, N] <- [3, 4] x [4, N]
    cam_coords = cam_coords.view(*pc.shape)  # [3, H_ds, W_ds] = [3, 60, 90]
    cam_coords = cam_coords * (pc[0] != -1) + pc[0] * (pc[0] == -1)  # [3, H_ds, W_ds] = [3, 60, 90]
    cam_coords = cam_coords.numpy().reshape(3, -1).transpose(1, 0)  # send in [N, 3] array
    cam_coords = cam_coords[mask_has_data]  # valid xyz, [X, 3]

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(cam_coords)
    pc_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    pc_o3d.normalize_normals()
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pc_o3d, [0, 0, 0])  # camera position is [0, 0, 0]

    normals_o3d = pc.numpy().reshape(3, -1).transpose(1, 0)  # [N, 3]
    normals_o3d[mask_has_data] = np.asarray(pc_o3d.normals)  # [X, 3]
    normals_o3d = normals_o3d.reshape(pc.size(1), pc.size(2), 3).transpose(2, 0, 1)  # [3, H_ds, W_ds]
    normals_o3d = torch.tensor(normals_o3d, dtype=torch.float)
    torch.save(normals_o3d, normal_path)

    with mp_lock:
        mp_counter.value += 1
        print("\rSurface normal extraction progress: {:d}/{:d}".format(mp_counter.value, ttl_size), end="", flush=True)


def _mp_func_pose(img_path, poses, txt_path, origin, mp_counter, mp_lock, ttl_size):
    """Multiprocessing backbone function for pose extraction"""
    str0, str1 = img_path.split('_')[1:3]
    str_idx = str1 if '-' in str0 else str0
    idx = int(str_idx)  # true index
    pose_gt = get_pose_mat(poses[idx])
    pose_gt[0:3, 3] -= origin
    np.savetxt(txt_path, pose_gt, newline='\n')
    with mp_lock:
        mp_counter.value += 1
        print("\rPose extraction progress: {:d}/{:d}".format(mp_counter.value, ttl_size), end="", flush=True)


def process_folder(src_cesium, src_other, dst, mode, origin):
    """Process raw data for each part per folder"""
    src_from = src_cesium if src_other is None else src_cesium
    print('\n===== Preparing {:s} data to {:s} in {:s} mode ====='.format(src_from, dst, mode))

    # prepare directories
    base_dir = os.path.join(dst, mode)

    image_cesium_ls = sorted(
        [os.path.join(src_cesium, path) for path in os.listdir(src_cesium) if path.endswith('_img.png')])
    if src_other is not None:
        if 'drone_translated' in mode:
            image_other_ls = sorted(
                [os.path.join(src_other, path) for path in os.listdir(src_other) if path.endswith('_fake_B.png')])
        elif 'drone_real' in mode:
            image_other_ls = sorted([os.path.join(src_other, path) for path in os.listdir(src_other) if
                                     path.endswith('.JPG') or path.endswith('.jpg')])
        else:
            raise NotImplementedError
        assert len(image_cesium_ls) == len(image_other_ls), 'Image size in translation and cesium set are not the same!'
    else:
        image_other_ls = None

    pc_ls = sorted([os.path.join(src_cesium, path) for path in os.listdir(src_cesium) if path.endswith('_pc.npy')])
    poses_npy = [path for path in os.listdir(src_cesium) if path.endswith('_poses.npy')]
    poses = np.load(os.path.join(src_cesium, poses_npy[0]))

    # multiprocessing utility
    mp_manger = mp.Manager()
    mp_counter = mp_manger.Value('i', 0)
    mp_lock = mp_manger.Lock()

    # follow DSAC* dataset convention
    sub_dirs = ['calibration', 'init', 'poses', 'rgb', 'depth', 'normal']
    for sub_dir in sub_dirs:
        cur_path = os.path.join(base_dir, sub_dir)
        mkdir(cur_path)
        if 'calibration' in sub_dir:
            # naively copy to txt files
            print('===== Preparing calibration folder ======')
            for _, img_path in enumerate(tqdm(image_cesium_ls)):
                txt_path = os.path.basename(img_path).replace('_img.png', '.txt')
                txt_path = os.path.join(cur_path, txt_path)
                with open(txt_path, 'w') as f:
                    f.write(str(focallength))
        elif 'init' in sub_dir:
            # downsample, shift the origin and save to torch-compatible binaries
            print('===== Preparing init folder ======')
            mp_func_args_ls = None
            with mp_lock:
                mp_counter.value = 0
            for _, pc_path in enumerate(tqdm(pc_ls)):
                dat_path = os.path.basename(pc_path)
                dat_src_path = os.path.join(src_cesium, dat_path)
                dat_path = os.path.join(cur_path, dat_path.replace('_pc.npy', '.dat'))

                # avoid duplication for non-synthetic matching datasets
                if mode == 'drone_sim' or mode == 'lhs_sim':
                    if mp_func_args_ls is None:
                        mp_func_args_ls = []
                    mp_func_args_ls.append((dat_src_path, dat_path, origin, mp_counter, mp_lock, len(pc_ls)))
                else:
                    src_path = os.path.join(dst, 'drone_sim', sub_dir, os.path.basename(dat_path))
                    assert os.path.exists(src_path)
                    target_path = dat_path
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.symlink(src_path, target_path)
            if mp_func_args_ls is not None:
                with mp.Pool() as pool:
                    pool.starmap(_mp_func_coor, mp_func_args_ls, chunksize=len(mp_func_args_ls)//10)
                print("")
        elif 'poses' in sub_dir:
            # shift the origin and save to txt
            print('===== Preparing poses folder ======')
            mp_func_args_ls = None
            with mp_lock:
                mp_counter.value = 0
            for _, img_path in enumerate(tqdm(image_cesium_ls)):
                img_path = os.path.basename(img_path)
                txt_path = os.path.basename(img_path).replace('_img.png', '.txt')
                txt_path = os.path.join(cur_path, txt_path)

                # avoid duplication for non-synthetic matching datasets
                if mode == 'drone_sim' or mode == 'lhs_sim':
                    if mp_func_args_ls is None:
                        mp_func_args_ls = []
                    mp_func_args_ls.append((img_path, poses, txt_path, origin, mp_counter, mp_lock, len(pc_ls)))
                else:
                    src_path = os.path.join(dst, 'drone_sim', sub_dir, os.path.basename(txt_path))
                    assert os.path.exists(src_path)
                    target_path = txt_path
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.symlink(src_path, target_path)
            if mp_func_args_ls is not None:
                with mp.Pool() as pool:
                    pool.starmap(_mp_func_pose, mp_func_args_ls, chunksize=len(mp_func_args_ls)//10)
                print("")
        elif 'rgb' in sub_dir:
            # use symbolic link to refer to actual dataset
            print('===== Preparing rgb folder ======')
            image_ls = image_cesium_ls if image_other_ls is None else image_other_ls
            for _, img_path in enumerate(tqdm(image_ls)):
                if mode == 'drone_sim' or mode == 'lhs_sim':
                    img_name_cesium = img_path
                elif mode == 'drone_translated' or mode == 'drone_real':
                    if mode == 'drone_translated':
                        key_name = os.path.basename(img_path).replace('_fake_B.png', '_img.png')
                    elif mode == 'drone_real':
                        key_name = os.path.basename(img_path).replace('.JPG', '_img.png')
                    img_name_cesium = []
                    for img_path_cesium in image_cesium_ls:
                        if key_name in img_path_cesium:
                            img_name_cesium.append(img_path_cesium)
                    assert len(img_name_cesium) == 1, 'More than one image is found in Cesium dataset!'
                    img_name_cesium = img_name_cesium[0]
                else:
                    raise NotImplementedError

                target_path = os.path.basename(img_name_cesium).replace('_img.png', '.png')
                target_path = os.path.join(cur_path, target_path)
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.symlink(img_path, target_path)
        elif 'depth' in sub_dir:
            # compute z-buffer depth from the down-sampled point cloud
            # MUST be called after init folder has been fully processed
            print('===== Preparing depth folder ======')
            mp_func_args_ls = None
            with mp_lock:
                mp_counter.value = 0
            for _, init_path in enumerate(tqdm(pc_ls)):
                init_src_path = os.path.join(base_dir, 'init', os.path.basename(init_path).replace('_pc.npy', '.dat'))
                depth_path = os.path.join(cur_path, '{:s}'.format(os.path.basename(init_src_path)))
                pose_txt_path = os.path.join(base_dir, 'poses', os.path.basename(init_src_path).replace('.dat', '.txt'))

                # avoid duplication for non-synthetic matching datasets
                if mode == 'drone_sim' or mode == 'lhs_sim':
                    if mp_func_args_ls is None:
                        mp_func_args_ls = []
                    mp_func_args_ls.append((init_src_path, depth_path, pose_txt_path, mp_counter, mp_lock, len(pc_ls)))
                else:
                    src_path = os.path.join(dst, 'drone_sim', sub_dir, os.path.basename(depth_path))
                    assert os.path.exists(src_path)
                    target_path = depth_path
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.symlink(src_path, target_path)
            if mp_func_args_ls is not None:
                with mp.Pool() as pool:
                    pool.starmap(_mp_func_depth, mp_func_args_ls, chunksize=len(mp_func_args_ls)//10)
                print("")
        elif 'normal' in sub_dir:
            # compute surface normals from the down-sampled point cloud
            # MUST be called after init folder has been fully processed
            print('===== Preparing normal folder ======')
            mp_func_args_ls = None
            with mp_lock:
                mp_counter.value = 0
            for _, init_path in enumerate(tqdm(pc_ls)):
                init_src_path = os.path.join(base_dir, 'init', os.path.basename(init_path).replace('_pc.npy', '.dat'))
                normal_path = os.path.join(cur_path, '{:s}'.format(os.path.basename(init_src_path)))
                pose_txt_path = os.path.join(base_dir, 'poses', os.path.basename(init_src_path).replace('.dat', '.txt'))

                # avoid duplication for non-synthetic matching datasets
                if mode == 'drone_sim' or mode == 'lhs_sim':
                    if mp_func_args_ls is None:
                        mp_func_args_ls = []
                    mp_func_args_ls.append((init_src_path, normal_path, pose_txt_path, mp_counter, mp_lock, len(pc_ls)))
                else:
                    src_path = os.path.join(dst, 'drone_sim', sub_dir, os.path.basename(normal_path))
                    assert os.path.exists(src_path)
                    target_path = normal_path
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.symlink(src_path, target_path)
            if mp_func_args_ls is not None:
                with mp.Pool() as pool:
                    pool.starmap(_mp_func_normal, mp_func_args_ls, chunksize=len(mp_func_args_ls)//10)
                print("")


def split_data(base_dir, mode, dataset_name):
    """split raw training data into training and validation parts"""
    # mode in ['lhs_sim', 'drone_sim', 'drone_translated', 'drone_real']
    symlink_re_register = False  # re-register symbolic links for some assets/sections
    nm_re_register_ls = None
    if mode == 'lhs_sim':
        split_ratio_ls = [0.9, 0.1]  # proportion of data for new split set
        nm_new_set_ls = ['train_sim', 'val_sim']  # folder name for new split set
        split_record = '{:s}_lhs_split.txt'.format(dataset_name)
    elif mode == 'drone_sim':  # trajectory-dataset generated from Cesium
        split_ratio_ls = [0.4, 0.1, 0.5]
        nm_new_set_ls = ['train_drone_sim', 'val_drone_sim', 'test_drone_sim']
        split_record = '{:s}_drone_split.txt'.format(dataset_name)
    elif mode == 'drone_translated':  # trajectory-dataset given by CycleGAN
        split_ratio_ls = [0.4, 0.1, 0.5]
        nm_new_set_ls = ['train_translated', 'val_translated', 'test_translated']
        split_record = '{:s}_drone_split.txt'.format(dataset_name)
        symlink_re_register = True
        nm_re_register_ls = ['train_drone_sim', 'val_drone_sim', 'test_drone_sim']
    elif mode == 'drone_real':  # trajectory-dataset taken by drone camera (w/ pre-processing)
        split_ratio_ls = [0.4, 0.1, 0.5]
        nm_new_set_ls = ['train_real', 'val_real', 'test_real']
        split_record = '{:s}_drone_split.txt'.format(dataset_name)
        symlink_re_register = True
        nm_re_register_ls = ['train_drone_sim', 'val_drone_sim', 'test_drone_sim']
    else:
        raise NotImplementedError

    # completeness check
    len_cal = len(os.listdir(os.path.join(base_dir, 'calibration')))
    len_init = len(os.listdir(os.path.join(base_dir, 'init')))
    len_poses = len(os.listdir(os.path.join(base_dir, 'poses')))
    len_rgb = len(os.listdir(os.path.join(base_dir, 'rgb')))
    assert len_cal == len_init and len_init == len_poses and len_poses == len_rgb
    dp_names = sorted([item.split('.')[0] for item in os.listdir(os.path.join(base_dir, 'rgb'))])

    # I/O of dataset split record
    split_record = os.path.abspath(os.path.join(os.path.dirname(__file__), split_record))
    if not os.path.exists(split_record):
        # Generate split record

        # create specific size for each subset
        size_new_set_ls = []
        for i, ratio in enumerate(split_ratio_ls):
            if i < len(split_ratio_ls) - 1:
                size_new_set_ls.append(int(len_cal * ratio))
            else:
                size_new_set_ls.append(int(len_cal - np.sum(size_new_set_ls)))
        assert np.sum(size_new_set_ls) == len_cal

        print('{:d} images are found in {:s}. Will be split into following subsets:'.format(len_cal, base_dir))
        [print("subset: {:s}, size: {:d}".format(nm, sz)) for nm, sz in zip(nm_new_set_ls, size_new_set_ls)]

        # generate subset indices randomly
        whole_idx_ls = np.random.permutation(np.arange(len_cal))
        split_section = np.cumsum(size_new_set_ls)
        idx_new_set_ls = np.split(whole_idx_ls, split_section)
        if len(idx_new_set_ls[-1]) == 0:
            idx_new_set_ls = idx_new_set_ls[:-1]
        idx_new_set_ls = [np.sort(sub_ls) for sub_ls in idx_new_set_ls]

        # sanity check
        assert len(np.concatenate(idx_new_set_ls)) == len_cal
        assert np.array_equal(np.sort(np.concatenate(idx_new_set_ls)), np.sort(whole_idx_ls))
        for idx_set_a, idx_set_b in itertools.combinations(idx_new_set_ls, 2):
            assert len(np.intersect1d(idx_set_a, idx_set_b)) == 0

        # write to split record
        filenames_new_set_ls = []
        with open(split_record, 'w') as f:
            for set_name, set_idx in zip(nm_new_set_ls, idx_new_set_ls):
                if set_name is None:
                    pass
                else:
                    section_name = set_name.split('_')[0]
                    f.write("Subset section ***{:s}***\n".format(section_name))
                    this_filenames = []
                    for idx in set_idx:
                        f.write(dp_names[idx] + '\n')
                        this_filenames.append(dp_names[idx])
                    filenames_new_set_ls.append(this_filenames)
    else:
        # Read from existing split record
        section_new_set_ls, idx_new_set_ls, this_idx_ls = [], [], []
        with open(split_record, 'r') as f:
            for line in f:
                if 'Subset section' in line:
                    assert line.split('***')[1] in nm_new_set_ls[len(section_new_set_ls)]
                    section_new_set_ls.append(line.split('***')[1])
                    if len(this_idx_ls):
                        idx_new_set_ls.append(np.asarray(this_idx_ls))
                        this_idx_ls = []
                else:
                    file_name = line.rstrip()
                    assert file_name in dp_names
                    this_idx_ls.append(dp_names.index(file_name))
            # append the last index list
            if len(this_idx_ls):
                idx_new_set_ls.append(np.asarray(this_idx_ls))

        # convert to filename
        filenames_new_set_ls = []
        for set_idx in idx_new_set_ls:
            this_filenames = []
            for idx in set_idx:
                this_filenames.append(dp_names[idx])
            filenames_new_set_ls.append(this_filenames)

        # sanity check
        assert len(np.concatenate(idx_new_set_ls)) == len_cal
        assert np.array_equal(np.sort(np.concatenate(idx_new_set_ls)), np.arange(len_cal))
        for idx_set_a, idx_set_b in itertools.combinations(idx_new_set_ls, 2):
            assert len(np.intersect1d(idx_set_a, idx_set_b)) == 0
        size_new_set_ls = [len(ls) for ls in idx_new_set_ls]
        assert np.sum(size_new_set_ls) == len_cal

        print('Successfully read preset file from {:s}'.format(split_record))
        print('{:d} images are found in {:s}. Will be split into following subsets:'.format(len_cal, base_dir))
        [print("subset: {:s}, size: {:d}".format(nm, sz)) for nm, sz in zip(nm_new_set_ls, size_new_set_ls)]

    # Move datasets
    sub_dirs = ['calibration', 'init', 'poses', 'rgb', 'depth', 'normal']
    for i, (set_name, set_filenames) in enumerate(zip(nm_new_set_ls, filenames_new_set_ls)):
        new_set = os.path.join(os.path.dirname(base_dir), set_name)
        mkdir(new_set)
        for folder in sub_dirs:
            if symlink_re_register and folder in ['init', 'poses', 'depth', 'normal']:
                re_register_path = os.path.abspath(os.path.join(os.path.dirname(base_dir), nm_re_register_ls[i], folder))
            else:
                re_register_path = None
            move_folder(os.path.join(base_dir, folder), os.path.join(new_set, folder), set_filenames, re_register_path)
        if 'DELETE' in set_name:
            shutil.rmtree(new_set)


def move_folder(source_dir, target_dir, filenames_new_set, re_register_path):
    """Move raw training data and do training-validation-testing split"""
    print('===== Processing {:s} folder ======'.format(source_dir))
    mkdir(target_dir)
    things = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
    for fp in tqdm(things):
        data_name = os.path.basename(fp)
        if data_name.split('.')[0] in filenames_new_set:  # remove extension
            if os.path.islink(fp) and not os.path.exists(fp):
                # re-register broken symbolic link
                os.remove(fp)
                os.symlink(os.path.join(re_register_path, os.path.basename(fp)), fp)
            shutil.move(fp, os.path.join(target_dir, data_name))
        else:
            pass


def virtual_merge_sections(src_dir_ls, dst_dir):
    """
    Construct virtual dataset pointer for some sections
    @param src_dir_ls: a list of section-wise paths
    @param dst_dir: destination path
    """
    print('===== Virtually merge {:d} folder ======'.format(len(src_dir_ls)))
    print("From:")
    for src_dir in src_dir_ls:
        print(src_dir)
    print("To: {:s}".format(dst_dir))

    mkdir(dst_dir)
    for src_dir in src_dir_ls:
        for sub_dir in ['calibration', 'init', 'poses', 'rgb', 'depth', 'normal']:
            mkdir(os.path.join(dst_dir, sub_dir))
            paths_from_ls = sorted(glob.glob(os.path.join(src_dir, sub_dir, '*')))
            paths_from_ls = [os.path.abspath(item) for item in paths_from_ls]
            for path_from in paths_from_ls:
                path_to = os.path.abspath(os.path.join(dst_dir, sub_dir, os.path.basename(path_from)))
                if os.path.exists(path_to):
                    os.remove(path_to)
                os.symlink(path_from, path_to)
