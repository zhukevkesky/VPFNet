from functools import partial

import spconv
import torch
import torch.nn as nn

from ...utils import common_utils
from ..backbones_3d.spconv_backbone import post_act_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self):
        super().__init__()
 
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

 
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submmm1'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submmm1'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submmm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconvvv2', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='submmm2'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='submmm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(128, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconvvv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='submmm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='submmm3'),
        )
 
 
 
        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(128, 128, indice_key='submmm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128*2, 128, 3, norm_fn=norm_fn, padding=1, indice_key='submmm3')
        self.inv_conv3 = block(128, 128, 3, norm_fn=norm_fn, indice_key='spconvvv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(128, 128, indice_key='submmm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(128*2, 128, 3, norm_fn=norm_fn, indice_key='submmm2')
        self.inv_conv2 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconvvv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(64, 64, indice_key='submmm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(64*2, 64, 3, norm_fn=norm_fn, indice_key='submmm1')

 
 

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    def UR_block_forward2(self, x_lateral, x_bottom, conv_t, conv_m):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def forward(self, x):
 
 
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
      
 
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward2(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1)
 
       
        return x_up1
 