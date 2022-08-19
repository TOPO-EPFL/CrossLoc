import torch
import torch.nn as nn
import numpy as np


def xyz2ae(xyz: torch.Tensor) -> torch.Tensor:
    """
    Turn normalized direction vector into azimuth and elevation.
    @param xyz  [B, 3, *] tensor of normalized direction vector.
    @return:    [B, 2, *] tensor of azimuth and elevation in radian.
    """

    # azimuth = arctan2(y, x), range [-pi, pi]
    azimuth = torch.atan2(xyz[:, 1], xyz[:, 0])  # [B, *]

    # elevation = arctan2(z, sqrt(x**2 + y**2)), range [-pi, pi]
    elevation = torch.atan2(xyz[:, 2], torch.norm(xyz[:, 0:2], dim=1, p=2))  # [B, *]

    return torch.stack([azimuth, elevation], dim=1)  # [B, 2, *]


def ae2xyz(ae: torch.Tensor) -> torch.Tensor:
    """
    Turn azimuth and elevation into normalized direction vector.
    @param ae   [B, 2, *] tensor of azimuth and elevation in radian.
    @return:    [B, 3, *] tensor of normalized direction vector.
    """
    XY_norm = torch.cos(ae[:, 1])  # [B, *]
    X = torch.cos(ae[:, 0]) * XY_norm  # [B, *]
    Y = torch.sin(ae[:, 0]) * XY_norm  # [B, *]
    Z = torch.sin(ae[:, 1])  # [B, *]
    XYZ = torch.stack([X, Y, Z], dim=1)  # [B, 3, *]
    return nn.functional.normalize(XYZ, p=2, dim=1)


def logits_to_radian(activation: torch.Tensor) -> torch.Tensor:
    """
    Convert the arbitrary activation into [-pi, pi] radian angle.
    @param activation: tensor of any size
    @return:
    """
    # radian = torch.tanh(activation) * np.pi
    radian = torch.sigmoid(activation).clamp(min=1.e-7, max=1 - 1.e-7)  # range [0, 1]
    radian = (radian * 2 - 1.0) * np.pi  # range [-pi, pi]
    return radian

