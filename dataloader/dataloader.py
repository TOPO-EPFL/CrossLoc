"""
Dataloader implementation based on DSAC* code.  (modified)
https://github.com/vislearn/dsacstar/blob/master/dataset.py

Contents of changes:
- Added grayscale option.
- Added RGBA to RGB image conversion.
- Added urbanscape & naturescape dataset statistics.
- Added mini-batch support (same scaling factor within one batch and it's reset per iteration).
- Added the option to return original RGB image.
- Added support for pre-computed depth & normal labels.
- Disabled reflection filling for the image rotational augmentation.
- Added support for multi-folder data source.
- Added mute option to be silent.
- Added support for fullsize 3D label and chunked real data.
- Added support for semantic label.

Copyright (c) 2020, Heidelberg University
Copyright (c) 2021, EPFL
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import pdb

import numpy as np
import random
import math

from skimage import io
from skimage import color
from skimage.transform import rotate, resize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from networks.networks import Network
from utils.io import safe_printout
from loss.semantics import trim_semantic_label


class CamLocDataset(Dataset):
    """
    Camera localization dataset.
    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(self, root_dir,
                 mode=1,
                 sparse=True,
                 coord=True,
                 depth=False,
                 normal=False,
                 semantics=False,
                 augment=False,
                 grayscale=False,
                 batch=True,
                 raw_image=False,
                 aug_rotation=30,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_contrast=0.1,
                 aug_brightness=0.1,
                 image_height=480,
                 real_data_chunk=None,
                 fullsize=False,
                 mute=False):
        """
        Constructor.
        Parameters:
            root_dir: Folder of the data (training or test).
            mode:
                0 = RGB only, load no initialization targets,
                1 = RGB + ground truth scene coordinates, load or generate ground truth scene coordinate targets
                2 = RGB-D, load camera coordinates instead of scene coordinates
            sparse: for mode = 1 (RGB+GT SC), load sparse initialization targets when True, load dense depth maps and generate initialization targets when False
            coord: Return 3D GT SC. (effective iff mode == 1 && sparse == True)
            depth: Return z-buffer depth. (effective iff mode == 1 && sparse == True)
            normal: Return surface normal. (effective iff mode == 1 && sparse == True)
            semantics: Return semantic labels. (effective iff mode == 1 && sparse == True)
            # Note: if more than one labels shall be retrieved, the output is a dictionary; see the end of __getitem__ for details
            augment: Use random data augmentation, note: not supported for mode = 2 (RGB-D) since pre-generateed eye coordinates cannot be agumented
            grayscale: Return grayscale or RGB image
            batch: Mini-batch mode (size > 1), affects the data augmentation during training
            raw_image: Return raw RGB image w/o any augmentation or normalization for post-processing
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions
            aug_scale_min: Lower limit of image scale factor for uniform sampling
            aug_scale_max: Upper limit of image scale factor for uniform sampling
            aug_contrast: Max relative scale factor for image contrast sampling, e.g. 0.1 -> [0.9,1.1]
            aug_brightness: Max relative scale factor for image brightness sampling, e.g. 0.1 -> [0.9,1.1]
            image_height: RGB images are rescaled to this maximum height
            real_data_chunk: the proportion of real images to load
            fullsize: to use full size 3D labels
            mute: to print out I/O-related message
        """
        self._config_dataloader(root_dir, mode, sparse, coord, depth, normal, semantics, augment, grayscale, batch,
                                raw_image, aug_rotation, aug_scale_min, aug_scale_max, aug_contrast, aug_brightness,
                                image_height, real_data_chunk, fullsize, mute)

    def _config_dataloader(self, root_dir, mode, sparse, coord, depth, normal, semantics, augment, grayscale, batch,
                           raw_image, aug_rotation, aug_scale_min, aug_scale_max, aug_contrast, aug_brightness,
                           image_height, real_data_chunk=None, fullsize=False, mute=False):
        """
        Re-usable backbone function for dataset configuration.
        """
        self.init = (mode == 1)
        self.sparse = sparse
        self.eye = (mode == 2)

        if self.init and self.sparse:
            self.coord = coord
            self.depth = depth
            self.normal = normal
            self.semantics = semantics
            if not np.any([coord, depth, normal, semantics]):
                raise Exception("At least one 3D label should be enabled! Coord: {}, Depth: {}, Normal: {}".format(
                    coord, depth, normal, semantics))
        else:
            self.coord = None
            self.depth = None
            self.normal = None
            self.semantics = None
        if not mute:
            print("Dataloader 3D label flags: coord: {}, depth: {}, normal: {}, semantics: {}".format(
                self.coord, self.depth, self.normal, self.semantics))

        self.image_height = image_height

        self.augment = augment
        self.grayscale = grayscale
        self.batch = batch
        self.raw_image = raw_image
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_contrast = aug_contrast
        self.aug_brightness = aug_brightness

        self.real_data_chunk = real_data_chunk
        if fullsize:
            Network.OUTPUT_SUBSAMPLE = 1
        self.fullsize = fullsize

        if self.eye and self.augment and (self.aug_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
            safe_printout("WARNING: Check your augmentation settings. Camera coordinates will not be augmented.")

        if self.grayscale:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_height),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    # mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                    # std=[0.25]
                    # urbanscape statistics (should generalize well enough)
                    mean=[0.4308],
                    std=[0.1724]
                    # naturescape statistics (backup)
                    # mean=[0.4084],
                    # std=[0.1404]
                )
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_height),
                transforms.ToTensor(),
                transforms.Normalize(
                    # urbanscape statistics (should generalize well enough)
                    mean=[0.4245, 0.4375, 0.3836],
                    std=[0.1823, 0.1701, 0.1854]
                    # naturescape statistics (backup)
                    # mean = [0.3636, 0.4331, 0.2956],
                    # std = [0.1383, 0.1457, 0.1147]
                )
            ])

        # Return raw RGB images w/o augmentation or normalization
        # Warning: this option superposes the other parameters, and should be used w/ care.
        if self.raw_image:
            self.augment = False
            self.grayscale = False
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_height),
                transforms.ToTensor()
            ])

        self.rgb_files = []
        self.pose_files = []
        self.calibration_files = []
        self.coord_files = []
        if self.depth:
            self.depth_files = []
        if self.normal:
            self.normal_files = []
        if self.semantics:
            self.semantics_files = []

        if isinstance(root_dir, list):
            root_dir_ls = root_dir
        elif os.path.isdir(root_dir):
            root_dir_ls = [root_dir]
        else:
            raise ValueError("root_dir type {} is not supported!".format(type(root_dir)))

        for base_dir in root_dir_ls:

            rgb_dir = base_dir + '/rgb/'
            pose_dir = base_dir + '/poses/'
            calibration_dir = base_dir + '/calibration/'
            if self.eye:
                coord_dir = base_dir + '/eye/'
            elif self.sparse:
                coord_dir = base_dir + '/init/'
            else:
                coord_dir = base_dir + '/depth/'
                assert not self.depth, "RGB+D mode is not compatible with depth label retrieval!"

            depth_dir = base_dir + '/depth/'
            normal_dir = base_dir + '/normal/'
            semantics_dir = base_dir + '/semantics/'

            _rgb_files = os.listdir(rgb_dir)
            _rgb_files = [rgb_dir + f for f in _rgb_files]
            _rgb_files.sort()
            self.rgb_files.extend(_rgb_files)

            _pose_files = os.listdir(pose_dir)
            _pose_files = [pose_dir + f for f in _pose_files]
            _pose_files.sort()
            self.pose_files.extend(_pose_files)

            _calibration_files = os.listdir(calibration_dir)
            _calibration_files = [calibration_dir + f for f in _calibration_files]
            _calibration_files.sort()
            self.calibration_files.extend(_calibration_files)

            if self.init or self.eye:
                _coord_files = os.listdir(coord_dir)
                _coord_files = [coord_dir + f for f in _coord_files]
                _coord_files.sort()
                self.coord_files.extend(_coord_files)

            if self.depth:
                _depth_files = os.listdir(depth_dir)
                _depth_files = [depth_dir + f for f in _depth_files]
                _depth_files.sort()
                self.depth_files.extend(_depth_files)

            if self.normal:
                _normal_files = os.listdir(normal_dir)
                _normal_files = [normal_dir + f for f in _normal_files]
                _normal_files.sort()
                self.normal_files.extend(_normal_files)

            if self.semantics:
                _semantics_files = os.listdir(semantics_dir)
                _semantics_files = [semantics_dir + f for f in _semantics_files]
                _semantics_files.sort()
                self.semantics_files.extend(_semantics_files)

        if len(self.rgb_files) != len(self.pose_files):
            raise Exception('RGB file count does not match pose file count!')

        if not sparse:

            # create grid of 2D pixel positions when generating scene coordinates from depth
            self.prediction_grid = np.zeros((2,
                                             math.ceil(1080 / Network.OUTPUT_SUBSAMPLE),
                                             math.ceil(1080 / Network.OUTPUT_SUBSAMPLE)))

            for x in range(0, self.prediction_grid.shape[2]):
                for y in range(0, self.prediction_grid.shape[1]):
                    self.prediction_grid[0, y, x] = x * Network.OUTPUT_SUBSAMPLE
                    self.prediction_grid[1, y, x] = y * Network.OUTPUT_SUBSAMPLE

    def __len__(self):
        return len(self.rgb_files)

    def _fetch_datapoint(self, idx):
        """
        Re-usable backbone function for mini-batch data retrieval.
        """
        image = io.imread(self.rgb_files[idx])

        if len(image.shape) < 3:
            image = color.gray2rgb(image)
        if len(image.shape) == 3 and image.shape[-1] == 4:
            # RGBA to RGB for Cesium dataset
            image = image[:, :, :3]

        focal_length = float(np.loadtxt(self.calibration_files[idx]))

        # image will be normalized to standard height, adjust focal length as well
        f_scale_factor = self.image_height / image.shape[0]
        focal_length *= f_scale_factor

        pose = np.loadtxt(self.pose_files[idx])
        pose = torch.from_numpy(pose).float()

        coords, depth, normal, semantics = 0, 0, 0, 0
        if self.init:
            if self.sparse:
                if self.coord:
                    coords = torch.load(self.coord_files[idx])  # [3, H_ds, W_ds]
                if self.depth:
                    depth = torch.load(self.depth_files[idx]).unsqueeze(0)  # [1, H_ds, W_ds]
                if self.normal:
                    normal = torch.load(self.normal_files[idx])  # [3, H_ds, W_ds]
                if self.semantics:
                    semantics = trim_semantic_label(np.load(self.semantics_files[idx]))
                    semantics = torch.tensor(semantics, dtype=torch.float).unsqueeze(0)  # [1, H_ds, W_ds]
            else:
                depth = io.imread(self.coord_files[idx])
                depth = depth.astype(np.float64)
                depth /= 1000  # from millimeters to meters
        elif self.eye:
            coords = torch.load(self.coord_files[idx])
        else:
            coords = 0

        if self.augment:

            # if mini-batch size is larger than 1, resizing is done in the collate_fn after the batch data is fetched.
            if self.batch:
                scale_factor = 1.0
                angle = 0.0
            else:
                scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
                angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            # augment input image
            if self.grayscale:
                cur_image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(int(self.image_height * scale_factor)),
                    transforms.Grayscale(),
                    transforms.ColorJitter(brightness=self.aug_brightness, contrast=self.aug_contrast),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # urbanscape statistics (should generalize well enough)
                        mean=[0.4308],
                        std=[0.1724]
                        # naturescape statistics
                        # mean=[0.4084],
                        # std=[0.1404]
                    )
                ])
            else:
                cur_image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(int(self.image_height * scale_factor)),
                    transforms.ColorJitter(brightness=self.aug_brightness, contrast=self.aug_contrast),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        # urbanscape statistics (should generalize well enough)
                        mean=[0.4245, 0.4375, 0.3836],
                        std=[0.1823, 0.1701, 0.1854]
                        # naturescape statistics
                        # mean = [0.3636, 0.4331, 0.2956],
                        # std = [0.1383, 0.1457, 0.1147]
                    )
                ])
            image = cur_image_transform(image)

            # scale focal length
            focal_length *= scale_factor

            # rotate input image
            def my_rot(t, angle, order, mode='constant'):
                t = t.permute(1, 2, 0).numpy()
                t = rotate(t, angle, order=order, mode=mode, cval=-1)
                t = torch.from_numpy(t).permute(2, 0, 1).float()
                return t

            image = my_rot(image, angle, 1, 'constant')

            if self.init:

                if self.sparse:
                    # rotate and scale initalization targets
                    coords_w = math.ceil(image.size(2) / Network.OUTPUT_SUBSAMPLE)
                    coords_h = math.ceil(image.size(1) / Network.OUTPUT_SUBSAMPLE)
                    if self.coord:
                        coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w), mode='nearest')[0]
                        coords = my_rot(coords, angle, 0)
                    if self.depth:
                        depth = F.interpolate(depth.unsqueeze(0), size=(coords_h, coords_w), mode='nearest')[0]
                        depth = my_rot(depth, angle, 0)
                    if self.normal:
                        normal = F.interpolate(normal.unsqueeze(0), size=(coords_h, coords_w), mode='nearest')[0]
                        normal = my_rot(normal, angle, 0)
                    if self.semantics:
                        # note: semantic map is always the same as image size
                        semantics = F.interpolate(semantics.unsqueeze(0), size=(image.size(1), image.size(2)),
                                                  mode='nearest')[0]
                        semantics = my_rot(semantics, angle, 0).clamp(min=0)

                else:
                    # rotate and scale depth maps
                    depth = resize(depth, image.shape[1:], order=0, cval=-1)
                    depth = rotate(depth, angle, order=0, mode='constant', cval=-1)

            # rotate ground truth camera pose
            angle = angle * math.pi / 180
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)

            pose = torch.matmul(pose, pose_rot)

        else:

            image = self.image_transform(image)

        if self.init and not self.sparse:
            # generate initialization targets from depth map

            offsetX = int(Network.OUTPUT_SUBSAMPLE / 2)
            offsetY = int(Network.OUTPUT_SUBSAMPLE / 2)

            coords = torch.zeros((
                3,
                math.ceil(image.shape[1] / Network.OUTPUT_SUBSAMPLE),
                math.ceil(image.shape[2] / Network.OUTPUT_SUBSAMPLE)))

            # subsample to network output size
            depth = depth[offsetY::Network.OUTPUT_SUBSAMPLE, offsetX::Network.OUTPUT_SUBSAMPLE]

            # construct x and y coordinates of camera coordinate
            xy = self.prediction_grid[:, :depth.shape[0], :depth.shape[1]].copy()
            # add random pixel shift
            xy[0] += offsetX
            xy[1] += offsetY
            # substract principal point (assume image center)
            xy[0] -= image.shape[2] / 2
            xy[1] -= image.shape[1] / 2
            # reproject
            xy /= focal_length
            xy[0] *= depth
            xy[1] *= depth

            # assemble camera coordinates trensor
            eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
            eye[0:2] = xy
            eye[2] = depth
            eye[3] = 1

            # eye to scene coordinates
            sc = np.matmul(pose.numpy(), eye.reshape(4, -1))
            sc = sc.reshape(4, depth.shape[0], depth.shape[1])

            # mind pixels with invalid depth
            sc[:, depth == 0] = 0
            sc[:, depth > 1000] = 0
            sc = torch.from_numpy(sc[0:3])

            coords[:, :sc.shape[1], :sc.shape[2]] = sc

        return image, pose, (coords, depth, normal, semantics), focal_length, self.rgb_files[idx]

    def __getitem__(self, idx):
        """
        Wrapper for dataloader single datapoint retrieval.
        """
        image, pose, (coords, depth, normal, semantics), focal_length, rgb_file = self._fetch_datapoint(idx)

        if self.init and self.sparse:
            flags = np.array([self.coord, self.depth, self.normal, self.semantics])
            if np.sum(flags) == 1:
                flag_idx = int(np.where(flags == True)[0])
                outputs = [coords, depth, normal, semantics][flag_idx]
            else:
                outputs = {"coord": coords, "depth": depth, "normal": normal, "semantics": semantics}
            return image, pose, outputs, focal_length, rgb_file
        else:
            return image, pose, coords, focal_length, rgb_file

    def batch_resize(self, batch):
        """
        Backbone collate_fn to resize data (images & coords) using a common scale factor.
        Usage: torch.utils.data.DataLoader(..., collate_fn=YOUR_DATASET.batch_resize, ...)
        """

        b_image = [item[0] for item in batch]
        b_pose = [item[1] for item in batch]
        b_geo_labels = [item[2] for item in batch]
        b_focal_length = [item[3] for item in batch]
        b_rgb_files = [item[4] for item in batch]

        if self.augment and self.batch:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            image_h = math.ceil(b_image[0].size(1) * scale_factor)
            image_w = math.ceil(b_image[0].size(2) * scale_factor)

            b_image_tensor = torch.stack(b_image, dim=0)
            b_image_tensor = F.interpolate(b_image_tensor, size=(image_h, image_w), mode='bilinear',
                                           align_corners=False)
            b_image_tensor = transforms.functional.rotate(b_image_tensor, angle, fill=-1)

            b_focal_length = [item * scale_factor for item in b_focal_length]

            if self.init or self.eye:
                # scale coordinates
                coords_w = math.ceil(b_image_tensor[0].size(2) / Network.OUTPUT_SUBSAMPLE)
                coords_h = math.ceil(b_image_tensor[0].size(1) / Network.OUTPUT_SUBSAMPLE)

                if isinstance(b_geo_labels[0], torch.Tensor):
                    # single 3d geometric label, such as coords/depth/normal
                    trim_b_geo_labels = torch.stack(b_geo_labels, dim=0)
                    if self.semantics:
                        trim_b_geo_labels = F.interpolate(trim_b_geo_labels, size=(image_h, image_w), mode='nearest')
                        trim_b_geo_labels = transforms.functional.rotate(trim_b_geo_labels, angle, fill=0)
                    else:
                        trim_b_geo_labels = F.interpolate(trim_b_geo_labels, size=(coords_h, coords_w), mode='nearest')
                        trim_b_geo_labels = transforms.functional.rotate(trim_b_geo_labels, angle, fill=-1)
                elif isinstance(b_geo_labels[0], dict):
                    # multiple 3d geometric labels, augment & concatenate tensors in one common dict
                    trim_b_geo_labels = {}
                    for key, value in zip(b_geo_labels[0].keys(), b_geo_labels[0].values()):
                        if isinstance(value, torch.Tensor):
                            b_data = torch.stack([item[key] for item in b_geo_labels], dim=0)
                            if self.semantics and key == 'semantics':
                                b_data = F.interpolate(b_data, size=(image_h, image_w), mode='nearest')
                                b_data = transforms.functional.rotate(b_data, angle, fill=0)
                            else:
                                b_data = F.interpolate(b_data, size=(coords_h, coords_w), mode='nearest')
                                b_data = transforms.functional.rotate(b_data, angle, fill=-1)
                            trim_b_geo_labels[key] = b_data
                        else:
                            trim_b_geo_labels[key] = 0
                else:
                    raise NotImplementedError
            else:
                trim_b_geo_labels = torch.stack(b_geo_labels, dim=0)

            return b_image_tensor, torch.stack(b_pose), trim_b_geo_labels, \
                   torch.tensor(b_focal_length, dtype=torch.float64), b_rgb_files

        else:
            if isinstance(b_geo_labels[0], torch.Tensor):
                # single 3d geometric label, such as coords/depth/normal/semantics
                trim_b_geo_labels = torch.stack(b_geo_labels, dim=0)
            elif isinstance(b_geo_labels[0], dict):
                # multiple 3d geometric labels, augment & concatenate tensors in one common dict
                trim_b_geo_labels = {}
                for key, value in zip(b_geo_labels[0].keys(), b_geo_labels[0].values()):
                    if isinstance(value, torch.Tensor):
                        b_data = torch.stack([item[key] for item in b_geo_labels], dim=0)
                        trim_b_geo_labels[key] = b_data
                    else:
                        trim_b_geo_labels[key] = 0
            else:
                raise NotImplementedError
            return torch.stack(b_image), torch.stack(b_pose), trim_b_geo_labels, \
                   torch.tensor(b_focal_length, dtype=torch.float64), b_rgb_files
