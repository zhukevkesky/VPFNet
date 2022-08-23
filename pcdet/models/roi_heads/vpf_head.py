
import torch.nn as nn
import torch
import numpy as np
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ..backbones_2d.submodule import DLASeg
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from ...datasets.augmentor.augmentor_utils import *
import spconv
from ...utils import torch_calib
from ...datasets.augmentor import augmentor_utils
from functools import partial




def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """

    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]   #### 100 , 90 
        x: (N)
        y: (N)

    Returns:

    """

    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + \
        torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'Transposeconv':
        conv = spconv.SparseConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class VPFHEAD(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        mlps2 = self.model_cfg.ROI_GRID_POOL.MLPS2
        mlps3 = self.model_cfg.ROI_GRID_POOL.MLPS3

        for k in range(len(mlps)):
            mlps[k] = [64] + mlps[k]
        for k in range(len(mlps2)):
            mlps2[k] = [64] + mlps2[k]

        for k in range(len(mlps3)):
            mlps3[k] = [64] + mlps3[k]

        self.roi_grid_pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
            query_ranges=self.model_cfg.ROI_GRID_POOL.QUERY_RANGES,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,

        )

        self.roi_grid_pool_layer2 = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(  # group_points_kernel_fast
            query_ranges=self.model_cfg.ROI_GRID_POOL.QUERY_RANGES2,
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS2,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE2,
            mlps=mlps2,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        self.pseudo_layer = voxelpool_stack_modules.NeighborStereoSAModuleMSG(
            query_ranges=self.model_cfg.ROI_GRID_POOL.QUERY_RANGES3,
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS3,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE3,
            mlps=mlps3,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE

        pre_channel = GRID_SIZE * GRID_SIZE * \
            GRID_SIZE * (32 * 2 + 32 * 2 + 32 * 2)

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(
                    pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        shared_fc_list3 = []
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * (32 * 2)
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list3.extend([
                nn.Conv1d(
                    pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list3.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer_ = nn.Sequential(*shared_fc_list)

        self.shared_fc_layer3_ = nn.Sequential(*shared_fc_list3)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        pre_channel = 256
        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )

        self.pseudo_cls = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )

        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.pseudo_reg = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.conv1 = spconv.SparseSequential(
            # post_act_block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='my_spconvv1', conv_type='spconv'),
            post_act_block(64, 64, 3, norm_fn=norm_fn,
                           padding=1, indice_key='submm1'),
            post_act_block(64, 64, 3, norm_fn=norm_fn,
                           padding=1, indice_key='submm1'),
            post_act_block(64, 64, 3, norm_fn=norm_fn,
                           padding=1, indice_key='submm1'),
            post_act_block(64, 64, 3, norm_fn=norm_fn,
                           padding=1, indice_key='submm1'),
            post_act_block(64, 64, 3, norm_fn=norm_fn,
                           padding=1, indice_key='submm1'),
        )

        self.dla34 = DLASeg()

        self.linear2 = nn.Sequential(
            nn.Linear(64,  64,  bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.norm = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):

        if weight_init == 'xavier':
            init_func = nn.init.xavier_normal_

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_layers[-1].bias, 0)

        nn.init.normal_(self.pseudo_reg[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.pseudo_reg[-1].bias, 0)
        nn.init.normal_(self.cls_layers[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.cls_layers[-1].bias, 0)
        nn.init.normal_(self.pseudo_cls[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.pseudo_cls[-1].bias, 0)

    def scatter_point_inds(self,  indices, point_inds, shape):
        ret = -1 * torch.ones(*shape, dtype=point_inds.dtype,
                              device=point_inds.device)
        ndim = indices.shape[-1]
        flattened_indices = indices.view(-1, ndim)
        slices = [flattened_indices[:, i] for i in range(ndim)]
        ret[slices] = point_inds
        return ret

    def generate_voxel2pinds(self, sparse_tensor):
        device = sparse_tensor.indices.device
        batch_size = sparse_tensor.batch_size
        spatial_shape = sparse_tensor.spatial_shape
        indices = sparse_tensor.indices.long()
        point_indices = torch.arange(
            indices.shape[0], device=device, dtype=torch.int32)
        output_shape = [batch_size] + list(spatial_shape)
        v2pinds_tensor = self.scatter_point_inds(
            indices, point_indices, output_shape)
        return v2pinds_tensor

    def tensor2points(self, tensor, offset=(0., -40., -3.), voxel_size=(.1, .1, .1)):
        indices = tensor.indices.float()

        offset = torch.Tensor(offset).to(indices.device)

        voxel_size = torch.Tensor(voxel_size).to(indices.device)
        indices[:, 1:] = indices[:, [3, 2, 1]] * \
            voxel_size + offset + .5 * voxel_size

        return tensor.features, indices

    def roi_grid_pseudo_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']

        P_rois = batch_dict['P_rois'].clone()

        left = batch_dict['left_img']
        right = batch_dict['right_img']
        calib = batch_dict['calib']
        h_shift = batch_dict['h_shift']
        operation = batch_dict['operation']

        x_conv3 = batch_dict['x_conv3']
        x_conv4 = batch_dict['x_conv4']

        X_resolution = 0.2
        Y_resolution = 0.2
        Z_resolution = 0.1

        f_conv3, v3_nxyz = self.tensor2points(tensor=x_conv3, offset=(
            0., -40., -3.), voxel_size=(.2, .2, .4))  # 0.05 , 0.05 , 0.1 4 倍
        f_conv4, v4_nxyz = self.tensor2points(tensor=x_conv4, offset=(
            0., -40., -3.), voxel_size=(.4, .4, .8))  # 0.05 , 0.05 , 0.1 8 倍

        global_roi_grid_points, _ = self.get_global_grid_points_of_roi(
            rois.clone(), grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )
        global_roi_grid_points = global_roi_grid_points.view(
            batch_size, -1, 4)  # (B, Nx6x6x6, 3)

        roi_grid_coords_x = (global_roi_grid_points[:, :, 0:1] - 0) // 0.05
        roi_grid_coords_y = (global_roi_grid_points[:, :, 1:2] - -40) // 0.05
        roi_grid_coords_z = (global_roi_grid_points[:, :, 2:3] - -3) // 0.1

        roi_grid_coords = torch.cat(
            [roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        xyz1 = v3_nxyz[:, 1:4]
        xyz_batch_cnt = xyz1.new_zeros(batch_size).int()
        batch_idx = v3_nxyz[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz1 = global_roi_grid_points.view(-1, 4)[:, :3]
        new_xyz_batch_cnt = xyz1.new_zeros(batch_size).int().fill_(
            global_roi_grid_points.shape[1])

        grid_batch_idx = rois.new_zeros(
            batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            grid_batch_idx[bs_idx, :, 0] = bs_idx
        cur_roi_grid_coords = roi_grid_coords // 4
        cur_roi_grid_coords = torch.cat(
            [grid_batch_idx, cur_roi_grid_coords], dim=-1)
        cur_roi_grid_coords = cur_roi_grid_coords.int()
        v2p_ind_tensor = self.generate_voxel2pinds(x_conv3)

        pooled_lidar_features = self.roi_grid_pool_layer(
            xyz=xyz1.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz1.contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=f_conv3.contiguous(),

            new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
            voxel2point_indices=v2p_ind_tensor
        )  # 7 ms

        xyz2 = v4_nxyz[:, 1:4]
        xyz_batch_cnt = xyz2.new_zeros(batch_size).int()
        batch_idx = v4_nxyz[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz2 = global_roi_grid_points.view(-1, 4)[:, :3]
        new_xyz_batch_cnt = xyz2.new_zeros(batch_size).int().fill_(
            global_roi_grid_points.shape[1])
        cur_roi_grid_coords2 = roi_grid_coords // 8
        cur_roi_grid_coords2 = torch.cat(
            [grid_batch_idx, cur_roi_grid_coords2], dim=-1)
        cur_roi_grid_coords2 = cur_roi_grid_coords2.int()
        v2p_ind_tensor2 = self.generate_voxel2pinds(x_conv4)

        pooled_lidar_features2 = self.roi_grid_pool_layer2(
            xyz=xyz2.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz2.contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=f_conv4.contiguous(),
            new_coords=cur_roi_grid_coords2.contiguous().view(-1, 4),
            voxel2point_indices=v2p_ind_tensor2
        )  # 6 ms

        pooled_lidar_features = pooled_lidar_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            32 * 2
        )  # (BxN, 6x6x6, C)

        pooled_lidar_features2 = pooled_lidar_features2.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            32 * 2
        )  # (BxN, 6x6x6, C)

        if True:

            roi_grid_coords_x2 = (
                global_roi_grid_points[:, :, 0:1] - 0) // (X_resolution)
            roi_grid_coords_y2 = (
                global_roi_grid_points[:, :, 1:2] - -40) // (Y_resolution)
            roi_grid_coords_z2 = (
                global_roi_grid_points[:, :, 2:3] - -3) // (Z_resolution)
            roi_grid_coords_z3 = global_roi_grid_points[:, :, 3:4]

            roi_grid_coords2 = torch.cat(
                [roi_grid_coords_x2, roi_grid_coords_y2, roi_grid_coords_z2, roi_grid_coords_z3], dim=-1)

            global_roi_grid_points3, global_roi_local3 = self.get_global_grid_points_of_roi3(
                P_rois.clone(), self.model_cfg.size_x, self.model_cfg.size_y,  self.model_cfg.size_z)  # (BxN, 20x20x20, 3)

            global_roi_grid_points3 = global_roi_grid_points3.view(
                batch_size, -1, 3)  # (B, Nx6x6x6, 3)
            global_roi_local3 = global_roi_local3.view(batch_size, -1, 3)

            cur_points = global_roi_grid_points3[:, :, :3].clone()

            cur_points[:, :, 0] = (cur_points[:, :, 0] - 0) / X_resolution
            cur_points[:, :, 1] = (cur_points[:, :, 1] - -40) / Y_resolution
            cur_points[:, :, 2] = (cur_points[:, :, 2] - -3) / Z_resolution

            cur_points = cur_points.floor()
            global_roi_local3 = global_roi_local3[:, :, :3]

            refimg_fea, _ = self.dla34(left)

            targetimg_fea, _ = self.dla34(right)

            voxel_features = []
            final_coords = []
            for i in range(len(calib)):

                mask = (cur_points[i, :, 0] > 0) & (cur_points[i, :, 0] < 350) & (cur_points[i, :, 1] < 400) & (cur_points[i, :, 1] > 0) \
                    & (cur_points[i, :, 2] < 40) & (cur_points[i, :, 2] > 0)

                cur_voxel_coords = cur_points[i][mask]
                cur_local_points = global_roi_local3[i][mask]

                calib_torch = torch_calib.torchCalib(calib[i], h_shift[i])

                trans_points = cur_voxel_coords.clone().float()
                trans_points[:, 0] = cur_voxel_coords[:, 0] * \
                    X_resolution + 0 + X_resolution / 2
                trans_points[:, 1] = cur_voxel_coords[:, 1] * \
                    Y_resolution - 40 + Y_resolution / 2
                trans_points[:, 2] = cur_voxel_coords[:, 2] * \
                    Z_resolution - 3 + Z_resolution / 2

                if self.training:
                    trans_points = augmentor_utils.p_scaling(
                        trans_points,  operation[i])
                    trans_points = augmentor_utils.p_rotation(
                        trans_points,  operation[i])
                    trans_points = augmentor_utils.p_flip(
                        trans_points, operation[i])

                c, w, h = refimg_fea[i].shape

                voxel_2d_left, _ = calib_torch.lidar_to_img_left(trans_points)
                voxel_2d_left = voxel_2d_left / 2

                voxel_2d_right, _ = calib_torch.lidar_to_img_right(
                    trans_points)
                voxel_2d_right = voxel_2d_right / 2

                mask = (voxel_2d_right[:, 0] < h - 1) & (voxel_2d_right[:,  0] >= 0) & \
                    (voxel_2d_right[:, 1] < w - 1) & (voxel_2d_right[:,  1] >= 0) & (voxel_2d_left[:, 0] < h - 1) & (voxel_2d_left[:,  0] >= 0) & \
                    (voxel_2d_left[:, 1] < w -
                     1) & (voxel_2d_left[:,  1] >= 0)

                voxel_2d_left = voxel_2d_left[mask]
                voxel_2d_right = voxel_2d_right[mask]
                trans_points = trans_points[mask]
                cur_voxel_coords = cur_voxel_coords[mask]
                cur_local_points = cur_local_points[mask]

                voxel_coords_additional = torch.zeros(
                    [cur_voxel_coords.shape[0], 1], device='cuda')
                voxel_coords_additional[:, 0] = i
                cur_voxel_coords = torch.cat(
                    [voxel_coords_additional, cur_voxel_coords], dim=1)

                cur_local_points = torch.cat(
                    [voxel_coords_additional, cur_local_points], dim=1)

                left_feature = refimg_fea[i].permute(1, 2, 0)
                channel_left = bilinear_interpolate_torch(
                    left_feature, voxel_2d_left[:, 0], voxel_2d_left[:, 1])

                right_feature = targetimg_fea[i].permute(1, 2, 0)
                channel_right = bilinear_interpolate_torch(
                    right_feature, voxel_2d_right[:, 0], voxel_2d_right[:, 1])

                channel = torch.cat(
                    [channel_left, channel_right], dim=1)  # 256  32

                cur_voxel_coords = cur_voxel_coords[:, [0, 3, 2, 1]]
                final_coords.append(cur_voxel_coords)
                voxel_features.append(channel)

            final_coords = torch.cat(final_coords, dim=0)
            voxel_features = torch.cat(voxel_features, dim=0)

            voxel_features = self.linear2(voxel_features)

            batch_size = batch_dict['batch_size']

            for i in range(batch_size):
                add_feature = torch.zeros([2, 64], device='cuda')
                add_xyz = torch.zeros([2, 4], device='cuda')
                add_xyz[0, 0] = i
                add_xyz[0, 1:4] = 4
                final_coords = torch.cat([final_coords, add_xyz], dim=0)
                voxel_features = torch.cat(
                    [voxel_features, add_feature], dim=0)

            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=final_coords.int(),
                spatial_shape=[40, 400, 350],
                batch_size=batch_size
            )

            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=final_coords.int(),
                spatial_shape=[40, 400, 350],
                batch_size=batch_size
            )

            p_conv2 = self.conv1(input_sp_tensor)

            X_resolution = 0.2
            Y_resolution = 0.2
            Z_resolution = 0.1

            dense_P_conv2, P_nxyz = self.tensor2points(tensor=p_conv2, offset=(
                0., -40., -3.), voxel_size=(X_resolution, Y_resolution, Z_resolution))

            cur_roi_grid_coords3 = roi_grid_coords2
            cur_roi_grid_coords3 = torch.cat(
                [grid_batch_idx, cur_roi_grid_coords3], dim=-1)
            cur_roi_grid_coords3 = cur_roi_grid_coords3.int()
            v2p_ind_tensor3 = self.generate_voxel2pinds(p_conv2)

            xyz3 = P_nxyz[:, 1:4]
            xyz_batch_cnt = xyz3.new_zeros(batch_size).int()
            batch_idx = P_nxyz[:, 0]
            for k in range(batch_size):
                xyz_batch_cnt[k] = (batch_idx == k).sum()

            new_xyz3 = global_roi_grid_points.view(-1, 4)[:, :3]
            new_xyz_batch_cnt = xyz3.new_zeros(batch_size).int().fill_(
                global_roi_grid_points.shape[1])

            pooled_pseudo_features = self.pseudo_layer(
                xyz=xyz3.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz3.contiguous(),
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=dense_P_conv2.contiguous(),
                new_coords=cur_roi_grid_coords3.contiguous().view(-1, 5),
                voxel2point_indices=v2p_ind_tensor3
            )  # (M1 + M2 ..., C)

            pooled_pseudo_features = pooled_pseudo_features.view(
                -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
                32 * 2
            )  # (BxN, 6x6x6, C)

        pooled_lidar_features3 = torch.cat(
            [pooled_lidar_features,  pooled_lidar_features2, pooled_pseudo_features], dim=2)

        batch_size_rcnn = pooled_lidar_features3.shape[0]
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE

        pooled_lidar_features3 = pooled_lidar_features3.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size,
                              grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        pooled_pseudo_features = pooled_pseudo_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size,
                              grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        fusion_feature = self.shared_fc_layer_(
            pooled_lidar_features3.view(batch_size_rcnn, -1, 1))

        if self.training:
            pooled_pseudo_features = self.shared_fc_layer3_(
                pooled_pseudo_features.view(batch_size_rcnn, -1, 1))  # B*N, 256, 1

        rcnn_cls = self.cls_layers(fusion_feature).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(fusion_feature).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if self.training:
            pseudo_cls = self.pseudo_cls(pooled_pseudo_features).transpose(
                1, 2).contiguous().squeeze(dim=1)
            pseudo_reg = self.pseudo_reg(pooled_pseudo_features).transpose(
                1, 2).contiguous().squeeze(dim=1)

            sp_time = 0
            seg_time = 0
        else:
            pseudo_cls = rcnn_cls
            pseudo_reg = rcnn_reg
            sp_time = 0
            seg_time = 0

        return rcnn_cls, rcnn_reg,   pseudo_reg, pseudo_cls, seg_time, sp_time

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(
            rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)

        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)

        global_center = rois[:, 0:3].clone()

        global_roi_grid_points += global_center.unsqueeze(dim=1)
        global_roi_grid_points = torch.cat(
            [global_roi_grid_points, local_roi_grid_points[:, :, 2:3]], dim=2)
        return global_roi_grid_points, local_roi_grid_points

    def get_global_grid_points_of_roi3(self, rois, size_x, size_y, size_z):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        rois[:, 3:5] += 0.8
        rois[:, 5] *= 1.3

        local_roi_grid_points = self.get_dense_grid_points2(
            rois, batch_size_rcnn, size_x, size_y, size_z)  # (B, 6x6x6, 3)

        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        )

        global_center = rois[:, 0:3].clone()
        local_roi_grid_points = global_roi_grid_points.clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points2(rois, batch_size_rcnn, size_x, size_y, size_z):
        faked_features = rois.new_ones((size_x, size_y, size_z))

        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]

        dense_idx = dense_idx.repeat(
            batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx.clone()

        roi_grid_points[:, :, 0] = (dense_idx[:, :, 0] + 0.5) / size_x * local_roi_size[:, 0].unsqueeze(dim=1) \
            - (local_roi_size[:, 0].unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        roi_grid_points[:, :, 1] = (dense_idx[:, :, 1] + 0.5) / size_y * local_roi_size[:, 1].unsqueeze(dim=1) \
            - (local_roi_size[:, 1].unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        roi_grid_points[:, :, 2] = (dense_idx[:, :, 2] + 0.5) / size_z * local_roi_size[:, 2].unsqueeze(dim=1) \
            - (local_roi_size[:, 2].unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)

        return roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(
            batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
            - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'], score_thres=self.model_cfg.NMS_CONFIG[
                'TRAIN' if self.training else 'TEST'].SCORE_THRESH
        )

        if self.training:

            for i in range(batch_dict['rois'].shape[0]):
                for j in range(batch_dict['rois'].shape[1]):
                    batch_dict['rois'][i, j,  6] += (
                        2 * self.model_cfg.rand_r * np.random.rand() - self.model_cfg.rand_r) * np.pi
                    batch_dict['rois'][i, j,  0] += (
                        2 * self.model_cfg.rand_xyz * np.random.rand() - self.model_cfg.rand_xyz)
                    batch_dict['rois'][i, j,  1] += (
                        2 * self.model_cfg.rand_xyz * np.random.rand() - self.model_cfg.rand_xyz)
                    batch_dict['rois'][i, j,  2] += (
                        2 * self.model_cfg.rand_xyz * np.random.rand() - self.model_cfg.rand_xyz)
                    batch_dict['rois'][i, j,  3] += (
                        2 * self.model_cfg.rand_whl * np.random.rand() - self.model_cfg.rand_whl)
                    batch_dict['rois'][i, j,  4] += (
                        2 * self.model_cfg.rand_whl * np.random.rand() - self.model_cfg.rand_whl)
                    batch_dict['rois'][i, j,  5] += (
                        2 * self.model_cfg.rand_whl * np.random.rand() - self.model_cfg.rand_whl)
                    batch_dict['rois'][i, j,
                                       6] = batch_dict['rois'][i, j, 6] % (2 * np.pi)

        if True:
            if self.training:
                targets_dict = self.assign_targets(batch_dict)

            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            if self.training:
                batch_dict['P_rois'] = targets_dict['P_rois'].clone()
            else:
                batch_dict['P_rois'] = batch_dict['rois'].clone()

            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling

        rcnn_cls, rcnn_reg, pseudo_reg, pseudo_cls, seg_time, sp_time = self.roi_grid_pseudo_pool(
            batch_dict)   # (BxN, 6x6x6, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg,

            )

            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_seg_time'] = seg_time
            batch_dict['batch_sp_time'] = sp_time
            batch_dict['batch_box_preds'] = batch_box_preds

            for index in range(batch_box_preds.shape[0]):

                for i in range(batch_box_preds.shape[1]):
                    if batch_dict['batch_box_preds'][index, i, :6].sum() == 0:
                        batch_dict['batch_cls_preds'][index, i] = -100

            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['pseudo_reg'] = pseudo_reg
            targets_dict['pseudo_cls'] = pseudo_cls

            self.forward_ret_dict = targets_dict

        return batch_dict
