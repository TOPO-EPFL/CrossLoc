import pdb

import torch
import dsacstar
import torch.nn as nn

from torch.multiprocessing import Pool


def set_device(device=None):
    """
    returns the current device if one exist
    else chooses best option (GPU > CPU)
    :param device:
    :return:
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    return device


def get_pose(sc, focal_length, image, hypotheses=64, threshold=10, inlieralpha=100, maxpixelerror=100, output_subsample=8,
             invalid_data=-1, use_nodata_vals=False):
    """
    Wrapper to calculate the pose using DSAC*. For context see https://github.com/vislearn/dsacstar/blob/master/test.py
    Default parameters as in options of DSAC*.
    :param sc: scene coordinates
    :param focal_length:
    :param image: image(for size)
    :param hypotheses:
    :param threshold:
    :param inlieralpha:
    :param maxpixelerror:
    :param output_subsample:
    :return: 4x4 pose (on cpu)
    """
    if (invalid_data != sc[0]).sum()//2 < inlieralpha:
        print('Dropping inlieralpha value ', (invalid_data != sc[0]).sum())
        inlieralpha = (invalid_data != sc[0]).sum()/2
        hypotheses = 10

    # enforce cpu location
    image = image.to('cpu')
    sc = sc.to('cpu')

    out_pose = torch.zeros((4, 4))

    dsacstar.forward_rgb(
        sc,
        out_pose,
        int(hypotheses),
        int(threshold),
        float(focal_length),
        float(image.size(3) / 2),  # principal point assumed in image center
        float(image.size(2) / 2),
        int(inlieralpha),
        int(maxpixelerror),
        int(output_subsample)
    )

    return out_pose


class Pose(nn.Module):
    """
    nn.Module wrapper to get the pose, speeds up data transfer for batch size > 1
    """
    def __init__(self, device=None):
        super(Pose, self).__init__()
        self.device = set_device(device)
        self.cpu = torch.device('cpu')
        # torch.multiprocessing.set_start_method('spawn')

    def forward(self, image, coords, focal_length):
        image = image.to(self.cpu)
        coords = coords.to(self.cpu)
        focal_length = focal_length.to(self.cpu)

        thread_in = [[coords[i].unsqueeze(0), float(focal_length[i]), image[i].unsqueeze(0)] for i in range(image.size(0))]

        with Pool(image.size(0)) as p:
            pose = torch.stack(p.map(self.multi_pose, thread_in), axis=0)
        # pose = torch.stack([self.multi_pose(sample) for sample in thread_in], axis=0)
        return pose.to(self.device)

    def multi_pose(self, x):
        return get_pose(x[0], x[1], x[2])
