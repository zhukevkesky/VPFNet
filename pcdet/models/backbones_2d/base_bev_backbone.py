import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
from functools import partial
 
from torch.nn.modules.batchnorm import _BatchNorm
 

 
def depth_to_color(depth):
    cmap = plt.cm.jet
    depth = np.max(abs(depth), axis=0)
    d_min = 0
    d_max = 15
    depth_relative = (depth - d_min)/(d_max-d_min)
    return 255 * cmap(depth_relative)[:, :, :3]


def depth_to_color2(depth):
    cmap = plt.cm.jet
    depth = np.max(abs(depth), axis=0)
    d_min = 0
    d_max = 1
    depth_relative = (depth - d_min)/(d_max-d_min)
    return 255 * cmap(depth_relative)[:, :, :3]


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
        assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES

        num_levels = len(layer_nums)
        c_in_list = [ 256  , 64 ]  
 
 
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx]  , num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),                
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[idx], num_upsample_filters[idx],
                        upsample_strides[idx],
                        stride=upsample_strides[idx], bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))

        self.num_bev_features = 256

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        spatial_features = data_dict['spatial_features']

        ups = []

        x = spatial_features

        x1 = self.blocks[0](x)

        ups.append(self.deblocks[0](x1))

        x2 = self.blocks[1](x1)

        ups.append(self.deblocks[1](x2))

        lidar_2d = torch.cat(ups, dim=1)
  
 
 
        data_dict['spatial_lidar_2d'] = lidar_2d
        data_dict['spatial_fusion_2d'] = lidar_2d
  

        return data_dict
