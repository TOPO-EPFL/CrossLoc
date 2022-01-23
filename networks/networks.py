"""
Dataloader implementation based on DSAC* code.  (modified)
https://github.com/vislearn/dsacstar/blob/master/network.py

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
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, PROJECT_DIR)
from utils.io import safe_printout


class Network(nn.Module):
    """
    DSAC* official network.
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is sub-sampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, tiny):
        """
        Constructor.
        """
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, (256, 128)[tiny], 3, 2, 1)

        self.res1_conv1 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)
        self.res1_conv2 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 1, 1, 0)
        self.res1_conv3 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)

        self.res2_conv1 = nn.Conv2d((256, 128)[tiny], (512, 128)[tiny], 3, 1, 1)
        self.res2_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res2_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 3, 1, 1)

        if not tiny:
            self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)

        self.res3_conv1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)

        self.fc1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc3 = nn.Conv2d((512, 128)[tiny], 3, 1, 1, 0)

        # learned scene coordinates relative to a mean coordinate (e.g. center of the scene)
        self.register_buffer('mean', torch.tensor(mean.size()).cuda())
        self.mean = mean.clone()
        self.tiny = tiny

    def forward(self, inputs):
        """
        Forward pass.

        inputs -- 4D data tensor (BxCxHxW)
        """

        x = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        if not self.tiny:
            res = self.res2_skip(res)

        res = res + x

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        sc[:, 0] += self.mean[0]
        sc[:, 1] += self.mean[1]
        sc[:, 2] += self.mean[2]

        return sc


def _create_res_block(tiny, num_gn_channel, ch_down_factor=1):
    """Create residual block"""
    num_ch = (512, 128)[tiny] // ch_down_factor
    res_block = nn.Sequential(nn.Conv2d(num_ch, num_ch, 3, 1, 1),
                              nn.GroupNorm(min(num_gn_channel, num_ch), num_ch),
                              nn.ReLU(),
                              nn.Conv2d(num_ch, num_ch, 1, 1, 0),
                              nn.GroupNorm(min(num_gn_channel, num_ch), num_ch),
                              nn.ReLU(),
                              nn.Conv2d(num_ch, num_ch, 3, 1, 1),
                              nn.GroupNorm(min(num_gn_channel, num_ch), num_ch),
                              nn.ReLU()
                              )
    return res_block


def _create_mlr_concatenator(num_mlr, tiny, num_gn_channel):
    """Create activation concatenation block for MLR."""
    in_channel = (512, 128)[tiny] * num_mlr
    out_channel = (512, 128)[tiny]
    mlr_block = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                              nn.GroupNorm(num_gn_channel, out_channel),
                              nn.ReLU(),
                              nn.Conv2d(out_channel, out_channel, 1, 1, 0),
                              nn.GroupNorm(num_gn_channel, out_channel),
                              nn.ReLU(),
                              nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                              nn.GroupNorm(num_gn_channel, out_channel),
                              nn.ReLU()
                              )
    return mlr_block


def _create_mlr_skip_layer(num_mlr, tiny, num_gn_channel):
    """Create skip layer for MLR"""
    in_channel = (512, 128)[tiny] * num_mlr
    out_channel = (512, 128)[tiny]
    skip_block = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, 0),
                               nn.GroupNorm(num_gn_channel, out_channel))
    return skip_block


class TransPoseNetEncoder(nn.Module):
    """A modular encoder for TransPose network."""
    def __init__(self, tiny, grayscale, enc_add_res_block=0, num_gn_channel=32):
        super(TransPoseNetEncoder, self).__init__()

        self.tiny = tiny
        self.grayscale = grayscale
        self.enc_add_res_block = enc_add_res_block
        self.num_gn_channel = num_gn_channel

        # Based on the DSAC* network (excluding FC layers)
        if grayscale:
            self.conv1 = nn.Conv2d(1, num_gn_channel, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(3, num_gn_channel, 3, 1, 1)
        self.norm1 = nn.GroupNorm(num_gn_channel, num_gn_channel)
        self.conv2 = nn.Conv2d(num_gn_channel, 64, 3, 2, 1)
        self.norm2 = nn.GroupNorm(num_gn_channel, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.norm3 = nn.GroupNorm(num_gn_channel, 128)
        self.conv4 = nn.Conv2d(128, (256, 128)[tiny], 3, 2, 1)
        self.norm4 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])

        self.res1_conv1 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)
        self.res1_norm1 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])
        self.res1_conv2 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 1, 1, 0)
        self.res1_norm2 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])
        self.res1_conv3 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)
        self.res1_norm3 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])

        self.res2_conv1 = nn.Conv2d((256, 128)[tiny], (512, 128)[tiny], 3, 1, 1)
        self.res2_norm1 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res2_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res2_norm2 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res2_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 3, 1, 1)
        self.res2_norm3 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])

        if not tiny:
            self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)
            self.res2_skip_norm = nn.GroupNorm(num_gn_channel, 512)

        # Additional residual block could be added on top of the vanilla encoder.
        self.enc_add_res_block_ls = [_create_res_block(tiny, num_gn_channel) for _ in range(enc_add_res_block)]
        for i, block in enumerate(self.enc_add_res_block_ls):
            self.add_module('enc_add_res_block{:d}'.format(i+1), block)

    def forward(self, inputs):
        """
        Forward pass.

        @param inputs           4D data tensor (BxCxHxW)
        """

        x = inputs

        """Encoder"""
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        res = F.relu(self.norm4(self.conv4(x)))

        x = F.relu(self.res1_norm1(self.res1_conv1(res)))
        x = F.relu(self.res1_norm2(self.res1_conv2(x)))
        x = F.relu(self.res1_norm3(self.res1_conv3(x)))

        res = F.relu(res + x)

        x = F.relu(self.res2_norm1(self.res2_conv1(res)))
        x = F.relu(self.res2_norm2(self.res2_conv2(x)))
        x = F.relu(self.res2_norm3(self.res2_conv3(x)))

        if not self.tiny:
            res = self.res2_skip_norm(self.res2_skip(res))

        res = F.relu(res + x)

        # additional residual block
        for i in range(len(self.enc_add_res_block_ls)):
            x = self.enc_add_res_block_ls[i](res)
            res = F.relu(res + x)

        return res


class DenseUpsamplingConvolution(nn.Module):
    """Modified dense upsampling convolution."""
    # Reference: https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/duc_hdc.py#L8
    def __init__(self, down_sampling_rate, in_channel, num_classes, num_gn_channel=32):
        super(DenseUpsamplingConvolution, self).__init__()
        up_sampling_channel = (down_sampling_rate ** 2) * num_classes
        self.conv = nn.Conv2d(in_channel, up_sampling_channel, 3, 1, 1)
        self.norm = nn.GroupNorm(num_gn_channel, up_sampling_channel)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_sampling_rate)

    def forward(self, x):
        x = self.relu(self.norm(self.conv(x)))
        x = self.pixel_shuffle(x)
        return x


class TransPoseNetDecoder(nn.Module):
    """A modular decoder for TransPose network."""
    def __init__(self, mean, tiny, dec_add_res_block=0,
                 num_task_channel=3, num_pos_channel=1, num_gn_channel=32, full_size_output=False):
        super(TransPoseNetDecoder, self).__init__()

        # learned output relative to its mean (e.g. center of the scene)
        self.register_buffer('mean', torch.tensor(mean.size()).cuda())
        self.mean = mean.clone()
        self.tiny = tiny
        self.dec_add_res_block = dec_add_res_block
        self.num_task_channel = num_task_channel
        self.num_pos_channel = num_pos_channel
        self.num_gn_channel = num_gn_channel
        self.full_size_output = full_size_output

        # Additional residual block could be added on top of the vanilla decoder.
        self.dec_add_res_block_ls = [_create_res_block(tiny, num_gn_channel) for _ in range(dec_add_res_block)]
        for i, block in enumerate(self.dec_add_res_block_ls):
            self.add_module('dec_add_res_block{:d}'.format(i+1), block)

        self.res3_conv1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_norm1 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res3_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_norm2 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res3_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_norm3 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])

        self.fc1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc1_norm = nn.GroupNorm(min((512, 128)[tiny], num_gn_channel), (512, 128)[tiny])
        self.fc2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc2_norm = nn.GroupNorm(min((512, 128)[tiny], num_gn_channel), (512, 128)[tiny])

        assert num_task_channel > 0 and num_pos_channel >= 0
        assert num_task_channel == len(mean)
        if full_size_output:
            # upsampling for semantics task
            self.duc_upsample = DenseUpsamplingConvolution(down_sampling_rate=8, in_channel=(512, 128)[tiny],
                                                           num_classes=num_task_channel+num_pos_channel)
            self.fc3 = nn.Conv2d(num_task_channel+num_pos_channel, num_task_channel+num_pos_channel, 1, 1, 0)
        else:
            self.fc3 = nn.Conv2d((512, 128)[tiny], num_task_channel + num_pos_channel, 1, 1, 0)

    def forward(self, inputs, up_height=None, up_width=None):
        """
        Forward pass.

        @param inputs           4D data tensor (BxCxHxW)
        @param up_height        Scalar, up-sampling target tensor height
        @param up_width         Scalar, up-sampling target tensor width
        """

        res = inputs

        # additional residual block
        # self.dec_add_res_block_ls[0][0] or self.res3_conv1 layer input is the intermediate activation [feature vec.].
        for i in range(len(self.dec_add_res_block_ls)):
            x = self.dec_add_res_block_ls[i](res)
            res = F.relu(res + x)

        x = F.relu(self.res3_norm1(self.res3_conv1(res)))
        x = F.relu(self.res3_norm2(self.res3_conv2(x)))
        x = F.relu(self.res3_norm3(self.res3_conv3(x)))

        res = F.relu(res + x)

        sc = F.relu(self.fc1_norm(self.fc1(res)))
        sc = F.relu(self.fc2_norm(self.fc2(sc)))
        if self.full_size_output:
            # upsampling for semantics task
            sc = self.duc_upsample(sc)  # [B, C, H', W']
            sc = F.interpolate(sc, (up_height, up_width), mode='bilinear', align_corners=False)  # trim dimensions

        sc = self.fc3(sc)

        sc[:, :self.num_task_channel] += self.mean[None, :, None, None]  # sc = [B, C, H, W], excluding positive channel

        if self.num_pos_channel:
            # constrain the torch.exp(*) output to be in [1.e-7, 1.e6]
            pos_output = F.hardtanh(sc[:, self.num_task_channel:].clone(),
                                    min_val=-16.10, max_val=13.82)  # auto-grad compatible clamp
            pos_output = torch.exp(pos_output)
            sc[:, self.num_task_channel:] = pos_output

        return sc


class TransPoseNet(nn.Module):
    """
    Flexible FCN architecture for various regression tasks.
    The output is sub-sampled by a factor of 8 compared to the image input.

    Contents of changes:
    - Added non-grayscale RGB image input.
    - Added group normalization.
    - Added encoder/decoder separation and supported an arbitrary number of residual blocks.
    - Added support for arbitrary-channel regression task output and positive-value uncertainty output.
    """

    def __init__(self, mean, tiny, grayscale, enc_add_res_block=0, dec_add_res_block=0,
                 num_task_channel=3, num_pos_channel=1, num_gn_channel=32,
                 num_mlr=0, num_unfrozen_encoder=0, full_size_output=False):
        """
        Constructor.
        @param mean                 Mean offset for task output.
        @param tiny                 Flag for tiny network.
        @param grayscale            Flag for grayscale image input.
        @param enc_add_res_block    Number of additional DSAC* style residual block for encoder.
        @param dec_add_res_block    Number of additional DSAC* style residual block for decoder.
        @param num_task_channel     Number of channels for underlying task.
        @param num_pos_channel      Number of channels for additional task w/ positive values, e.g., uncertainty.
        @param num_gn_channel       Number of group normalization channels, a hyper-parameter.
        @param num_mlr              Number of homogeneous mid-level representations encoders.
        @param num_unfrozen_encoder Number of encoders that are not frozen.
        @param full_size_output     Flag for full-size network output (by using DUC-style layers).

        Note: if enc_add_res_block == dec_add_res_block == 0 && num_task_channel == 3 && num_pos_channel = 0,
        the model become DSAC* net + group normalization only.
        """
        super(TransPoseNet, self).__init__()

        """Init"""
        # learned output relative to its mean (e.g. center of the scene)
        self.register_buffer('mean', torch.tensor(mean.size()).cuda())
        self.mean = mean.clone()
        self.tiny = tiny
        self.grayscale = grayscale
        self.enc_add_res_block = enc_add_res_block
        self.dec_add_res_block = dec_add_res_block
        self.num_task_channel = num_task_channel
        self.num_pos_channel = num_pos_channel
        self.num_gn_channel = num_gn_channel
        self.num_mlr = num_mlr
        self.full_size_output = full_size_output

        self.OUTPUT_SUBSAMPLE = 1 if full_size_output else 8

        """Vanilla encoder"""
        if num_mlr == 0:
            self.encoder = TransPoseNetEncoder(tiny, grayscale, enc_add_res_block, num_gn_channel)
            self.encoder_ls = [self.encoder]
        else:
            self.encoder = nn.Identity()
            self.encoder_ls = [self.encoder]

        """MLR encoders"""
        if num_mlr > 0 and isinstance(num_mlr, int):
            assert 0 <= num_unfrozen_encoder <= num_mlr
            self.mlr_encoder_ls = [TransPoseNetEncoder(tiny, grayscale, enc_add_res_block, num_gn_channel) for _ in range(num_mlr)]
            # Freeze gradients of the re-used encoder
            for i, block in enumerate(self.mlr_encoder_ls):
                if i >= num_unfrozen_encoder:  # the first few encoders **may** be reused for training
                    for param in block.parameters():
                        param.requires_grad = False
                self.add_module('mlr_encoder_{:d}'.format(i + 1), block)
            self.mlr_norm = nn.GroupNorm(num_gn_channel, (512, 128)[tiny] * num_mlr)
            self.mlr_forward = _create_mlr_concatenator(num_mlr, tiny, num_gn_channel)
            self.mlr_skip = _create_mlr_skip_layer(num_mlr, tiny, num_gn_channel)  # normalization is included
        else:
            self.mlr_encoder_ls = [nn.Identity()]
            self.mlr_norm = nn.Identity()
            self.mlr_forward = nn.Identity()
            self.mlr_skip = nn.Identity()
        self.mlr_ls = self.mlr_encoder_ls + [self.mlr_norm, self.mlr_forward, self.mlr_skip]

        """Decoder"""
        # we always have a decoder regardless of the #MLR
        self.decoder = TransPoseNetDecoder(mean, tiny, dec_add_res_block,
                                           num_task_channel, num_pos_channel, num_gn_channel, full_size_output)
        self.decoder_ls = [self.decoder]

        """Print out"""
        safe_printout('Initialized network w/ group normalization, Tiny net: {}, Grayscale input: {}, Fullsize output: {}.'.format(
            self.tiny, self.grayscale, self.full_size_output))
        safe_printout('#Aadditional residual blocks: Encoder: {:d}, Decoder: {:d}'.format(
            self.enc_add_res_block, self.dec_add_res_block))
        safe_printout('#Task output channel {:d}, #Positive-value output channel {:d}, #Group normalization channel: {:d}.'.format(
            self.num_task_channel, self.num_pos_channel, self.num_gn_channel))
        safe_printout('#MLR: {:d}, #Unfrozen encoder: {:d}'.format(num_mlr, num_unfrozen_encoder))

        ttl_num_param = 0
        param_info = 'Separation of #trainable parameters: '
        for name, struct in zip(['Vanilla encoder', 'MLR encoder', 'Decoder'],
                                [self.encoder_ls, self.mlr_ls, self.decoder_ls]):
            num_param = sum([param.numel() for layer in struct for param in layer.parameters() if param.requires_grad])
            ttl_num_param += num_param
            param_info += '{:s}: {:,d}, '.format(name, num_param)
        param_info += 'Total: {:,d}.'.format(ttl_num_param)
        safe_printout(param_info)

    def forward(self, inputs):
        """
        Forward pass.

        @param inputs           4D data tensor (BxCxHxW)
        """

        x = inputs
        up_height, up_width = inputs.size()[2:4]

        """Vanilla encoder"""
        if self.num_mlr == 0:
            res = self.encoder(x)
        else:
            res = None

        """MLR encoder"""
        if self.num_mlr:
            # inference
            mlr_activation_ls = [mlr_enc(inputs) for mlr_enc in self.mlr_encoder_ls]

            # activation concatenation
            mlr = torch.cat(mlr_activation_ls, dim=1)  # [B, C * #MLR, H, W]

            # forward
            res = self.mlr_skip(mlr)
            mlr = self.mlr_norm(mlr)
            mlr = self.mlr_forward(mlr)
            res = F.relu(res + mlr)

        """Decoder"""
        if self.full_size_output:
            sc = self.decoder(res, up_height, up_width)
        else:
            sc = self.decoder(res)

        return sc


class ProjHead(nn.Module):
    """
    Projection head for fixed-size feature vector extraction.
    """
    def __init__(self, in_channel, out_length=2048, tiny=False, num_gn_channel=32):
        super(ProjHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, (512, 128)[tiny], 3, 2, 1)
        self.norm1 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 3, 2, 1)
        self.norm2 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 3, 2, 1)
        self.norm3 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.conv4 = nn.Conv2d((512, 128)[tiny], out_length, 1, 1, 0)
        self.norm4 = nn.GroupNorm(num_gn_channel, out_length)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.in_channel = in_channel
        self.out_length = out_length
        self.num_gn_channel = num_gn_channel

        safe_printout('Initialized projection head w/ group normalization, #Input channels: {:d}, Output feature vector length: {:d}, #Group normalization channel: {:d}.'.format(
            self.in_channel, self.out_length, self.num_gn_channel))

        ttl_num_param = sum([param.numel() for param in self.parameters() if param.requires_grad])
        safe_printout('Projection head total number of trainable parameters: {:,d}'.format(ttl_num_param))

    def forward(self, inputs):
        """
        @inputs [B, C, H, W], H=60, W=90 if no data augmentation.
        """
        x = F.relu(self.norm1(self.conv1(inputs)))      # output: [B, 512, 30, 45]
        x = F.relu(self.norm2(self.conv2(x)))           # output: [B, 512, 15, 22]
        x = F.relu(self.norm3(self.conv3(x)))           # output: [B, 512, 7, 11]
        x = F.relu(self.norm4(self.conv4(x)))           # output: [B, out_length, 7, 11]
        x = self.avgpool(x)                             # output: [B, out_length, 1, 1]
        x = torch.flatten(x, 1)                         # output: [B, out_length]
        return x


