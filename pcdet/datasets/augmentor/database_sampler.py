import pickle
import torch.nn.functional as F
import numpy as np
import torch
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
import os
from PIL import Image
from .transforms import RandomColor, ToTensor, Compose, Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(
                    infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        self.occlu_thres_iou = sampler_cfg.get('occlu_thres_iou', 0)
        self.filter = sampler_cfg.get('filter', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

        train_transform_list = [
            RandomColor(),
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
        self.trans = Compose(train_transform_list)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]

            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (
                    key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group, calib_R0):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(
            sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0
        sampled_dict = []
        for idx in indices[pointer: pointer + sample_num]:
            sampled_dict.append(self.db_infos[class_name][idx])

        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]

        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        difficulty = data_dict['gt_difficulty'][gt_boxes_mask]

        image_2 = data_dict['left_img']
        image_3 = data_dict['right_img']
        snake = data_dict['snake']
        snake[:, :] = 0

        img_shape = data_dict['image_shape']

        image_2 = np.array(image_2)
        image_3 = np.array(image_3)
        calib = data_dict['calib']

        # project 3D GT boxes to 2D box
        sampled_gt_before = box_utils.boxes3d_lidar_to_kitti_camera(
            sampled_gt_boxes.copy(), calib)
        sampled_gt_before_left = box_utils.boxes3d_kitti_camera_to_imageboxes(
            sampled_gt_before, calib, image_shape=img_shape)
        sampled_gt_before_right = box_utils.boxes3d_kitti_camera_to_imageboxes_right(
            sampled_gt_before, calib, image_shape=img_shape)

        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes_new, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('road_plane')

        obj_points_list = []

        for idx, info in enumerate(total_valid_sampled_dict):
            lidar_path = self.root_path / info['path']
            folder = os.path.splitext(info['path'])[0]

            left_folder = folder + '_left.png'
            left_path = self.root_path / left_folder

            right_folder = folder + '_right.png'
            right_path = self.root_path / right_folder

            image2_obj = Image.open(left_path).convert('RGB')
            image3_obj = Image.open(right_path).convert('RGB')
            obj_points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            pts_2d, _ = calib.lidar_to_img_left(obj_points[:, :3])
            min_location_before = np.min(pts_2d[:, 1])

            pts_2d, _ = calib.lidar_to_img_right(obj_points[:, :3])
            min_location_before_right = np.min(pts_2d[:, 1])

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                obj_points[:, 2] = obj_points[:, 2] - mv_height[idx]

            pts_2d, _ = calib.lidar_to_img_left(obj_points[:, :3])
            min_location_after = np.min(pts_2d[:, 1])
            pts_2d, _ = calib.lidar_to_img_right(obj_points[:, :3])
            min_location_after_right = np.min(pts_2d[:, 1])

            # move the pasted points to the ground plane , then calculate the 2D offset
            mv_2d_height = min_location_after - min_location_before
            mv_2d_height = np.round(mv_2d_height)
            mv_2d_height_right = min_location_after_right - min_location_before_right
            mv_2d_height_right = np.round(mv_2d_height_right)

            # reshape the object patch, to fit the image size
            left_x, left_down, left_y, left_up = sampled_gt_before_left[idx, :]

            left_down = np.floor(left_down)
            left_up = np.floor(left_up)

            left_down += mv_2d_height
            left_up += mv_2d_height
            left_x = int(np.floor(left_x))
            left_down = int(np.floor(left_down))
            left_y = int(np.floor(left_y))
            left_up = int(np.floor(left_up))

            image2_obj = np.array(image2_obj)

            real_width = image2_obj.shape[1]

            if left_x == 0:
                obj_width = left_y - left_x
                image2_obj = image2_obj[:, int(obj_width - real_width):, :]

            elif left_y == image_2.shape[1]:
                obj_width = left_y - left_x
                image2_obj = image2_obj[:, :obj_width, :]

            else:
                obj_width = left_y - left_x
                image2_obj = image2_obj[:, :obj_width, :]

            real_height = image2_obj.shape[0]
            if left_down <= 0:
                left_down = 0
                obj_height = left_up - left_down
                image2_obj = image2_obj[int(real_height - obj_height):, :, :]
            elif left_up >= image_2.shape[0]:
                left_up = image_2.shape[0]
                obj_height = left_up - left_down
                image2_obj = image2_obj[:obj_height, :, :]

            else:
                obj_height = left_up - left_down
                image2_obj = image2_obj[:obj_height, :, :]

            o_height, o_width, _ = image2_obj.shape
            image_patch = image_2[left_down:int(
                left_down+o_height), left_x:int(left_x+o_width), :].copy()
            # paste the patch to the image
            n2 = np.mean(image2_obj, axis=2)
            mask_L = n2 > 0

            image_patch[mask_L] = image2_obj[mask_L].copy()
            image_2[left_down:int(
                left_down+o_height), left_x:int(left_x+o_width), :] = image_patch.copy()

            # reshape the object patch, to fit the image size
            left_x, left_down, left_y, left_up = sampled_gt_before_right[idx, :]
            left_down = np.floor(left_down)
            left_up = np.floor(left_up)
            left_down += mv_2d_height
            left_up += mv_2d_height

            left_x = int(np.floor(left_x))
            left_down = int(np.floor(left_down))
            left_y = int(np.floor(left_y))
            left_up = int(np.floor(left_up))

            image3_obj = np.array(image3_obj)

            real_width = image3_obj.shape[1]

            if left_x == 0:
                obj_width = left_y - left_x
                image3_obj = image3_obj[:, int(obj_width - real_width):, :]
            elif left_y == image_2.shape[1]:
                obj_width = left_y - left_x
                image3_obj = image3_obj[:, :obj_width, :]
            else:
                obj_width = left_y - left_x
                image3_obj = image3_obj[:, :obj_width, :]

            real_height = image3_obj.shape[0]
            if left_down <= 0:
                print(real_height, obj_height, left_up, left_down)
                left_down = 0
                obj_height = left_up - left_down

                image3_obj = image3_obj[int(real_height - obj_height):, :, :]
            elif left_up >= image_3.shape[0]:
                left_up = image_3.shape[0]
                obj_height = left_up - left_down
                image3_obj = image3_obj[:obj_height, :, :]
            else:
                obj_height = left_up - left_down
                image3_obj = image3_obj[:obj_height, :, :]

            o_height, o_width, _ = image3_obj.shape

            image_patch = image_3[left_down:int(
                left_down+o_height), left_x:int(left_x+o_width), :].copy()

            n3 = np.mean(image3_obj, axis=2)
            mask_R = n3 > 0
            image_patch[mask_R] = image3_obj[mask_R].copy()

            # paste the patch to the image

            image_3[left_down:int(
                left_down+o_height), left_x:int(left_x+o_width), :] = image_patch.copy()
            image3_obj = np.array(image3_obj)

            obj_points_list.append(obj_points)

        obj_points_all = np.concatenate(obj_points_list, axis=0)

        sampled_gt_names = np.array([x['name']
                                    for x in total_valid_sampled_dict])
        sampled_gt_diff = np.array([x['difficulty']
                                   for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes_new[:, 0:7].copy(), extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )

        points = box_utils.remove_points_in_boxes3d(
            points, large_sampled_gt_boxes)

        points = np.concatenate([obj_points_all, points], axis=0)

        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes_new], axis=0)
        difficulty = np.concatenate([difficulty, sampled_gt_diff], axis=0)

        data_dict['gt_difficulty'] = difficulty

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names

        image_2 = Image.fromarray(image_2)
        image_3 = Image.fromarray(image_3)

        data_dict['points'] = points

        data_dict['left_img'] = image_2
        data_dict['right_img'] = image_3
        data_dict['snake'] = snake

        return data_dict

    def image_box_overlap(self, boxes, query_boxes, criterion=-1):
        N = boxes.shape[0]
        K = query_boxes.shape[0]
        overlaps = np.zeros((N, K), dtype=boxes.dtype)
        for k in range(K):
            qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                         (query_boxes[k, 3] - query_boxes[k, 1]))
            for n in range(N):
                iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                      max(boxes[n, 0], query_boxes[k, 0]))
                if iw > 0:
                    ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                          max(boxes[n, 1], query_boxes[k, 1]))
                    if ih > 0:
                        if criterion == -1:
                            ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)

                        elif criterion == 0:
                            ua = ((boxes[n, 2] - boxes[n, 0]) *
                                  (boxes[n, 3] - boxes[n, 1]))
                        elif criterion == 1:
                            ua = qbox_area
                        elif criterion == 2:
                            ua = min((boxes[n, 2] - boxes[n, 0]) *
                                     (boxes[n, 3] - boxes[n, 1]),  qbox_area)

                        overlaps[n, k] = iw * ih / ua
        return overlaps

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        img_shape = data_dict['image_shape']

        calib = data_dict['calib']

        existed_boxes = gt_boxes
        left_img = data_dict['left_img']

        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():

            if int(sample_group['sample_num']) > 0:

                sampled_dict = self.sample_with_fixed_number(
                    class_name, sample_group, calib.R0[0, 2])

                sampled_boxes = np.stack(
                    [x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                cur_gt_box = data_dict['gt_boxes']

                cur_2d_box = box_utils.boxes3d_lidar_to_kitti_camera(
                    cur_gt_box.copy(), calib)
                existed_2d_box_left = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    cur_2d_box, calib, image_shape=img_shape)

                gt_existed_2d_box_left = existed_2d_box_left.copy()

                # init some boxes out of the detection range.
                existed_2d_box_left[:, :3] = 100
                existed_2d_box_left[:, 3:] = 0.1
                NNN = existed_2d_box_left.shape[0]

                for i in range(sampled_boxes.shape[0]):
                    use_sample = True

                    iou_calculate_box = sampled_boxes[i:i+1].copy()

                    mv_ground_snake_box_left = sampled_dict[i]['inmodal_box2'].copy(
                    )
                    mv_ground_snake_box_left = mv_ground_snake_box_left[np.newaxis, :]

                    mv_ground_box = box_utils.enlarge_height(
                        iou_calculate_box[:, 0:7].copy(),  extra_1=0, extra_2=0, extra_width=0
                    )
                    mv_ground_box = mv_ground_box.numpy()

                    mv_ground_box_2d = box_utils.boxes3d_lidar_to_kitti_camera(
                        mv_ground_box.copy(), calib)
                    mv_ground_box_2d_left = box_utils.boxes3d_kitti_camera_to_imageboxes(
                        mv_ground_box_2d, calib, image_shape=img_shape)

                    mv_ground_box_new = mv_ground_box

                    mv_ground_box_new_2d = box_utils.boxes3d_lidar_to_kitti_camera(
                        mv_ground_box_new.copy(), calib)
                    mv_ground_box_new_2d_left = box_utils.boxes3d_kitti_camera_to_imageboxes(
                        mv_ground_box_new_2d, calib, image_shape=img_shape)
                    mv_2d_height = mv_ground_box_new_2d_left[:,
                                                             1] - mv_ground_box_2d_left[:, 1]

                    if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                        mv_ground_snake_box_left[:,
                                                 1] = mv_ground_snake_box_left[:, 1] + mv_2d_height
                        mv_ground_snake_box_left[:,
                                                 3] = mv_ground_snake_box_left[:, 3] + mv_2d_height

                    # calculated collision IOUs
                    iou1 = self.image_box_overlap(
                        mv_ground_snake_box_left,  existed_2d_box_left, 2)
                    iou0 = self.image_box_overlap(
                        mv_ground_snake_box_left,  gt_existed_2d_box_left, 2)
                    iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(
                        iou_calculate_box[:, 0:7].copy(), existed_boxes[:, 0:7])

                    # filter the objects according to their collision IOUs
                    for j in range(NNN - 1,  iou1.shape[1]):

                        if iou1[0, j] > self.occlu_thres_iou:
                            use_sample = False

                    if (mv_ground_box_2d_left[0, 2] - mv_ground_box_2d_left[0, 0]) * (mv_ground_box_2d_left[0, 3] - mv_ground_box_2d_left[0, 1]) < 100 and class_name == 'Car':
                        use_sample = False

                    if (mv_ground_box_2d_left[0, 2] - mv_ground_box_2d_left[0, 0]) * (mv_ground_box_2d_left[0, 3] - mv_ground_box_2d_left[0, 1]) < 20:
                        use_sample = False

                    for j in range(iou0.shape[1]):

                        if iou0[0, j] > 0.2:
                            use_sample = False

                    for j in range(iou2.shape[1]):
                        if iou2[0, j] > 0:
                            use_sample = False

                    if use_sample == True:
                        valid_sampled_dict = sampled_dict[i:i+1]
                        cur_3D_box = iou_calculate_box
                        existed_boxes = np.concatenate(
                            (existed_boxes, cur_3D_box), axis=0)
                        existed_2d_box_left = np.concatenate(
                            (existed_2d_box_left,  mv_ground_snake_box_left), axis=0)
                        total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        # reorder the images, according to occlusion
        for j in range(sampled_gt_boxes.__len__()):

            max_num = sampled_gt_boxes[j, 0]

            for k in range(j + 1, sampled_gt_boxes.__len__()):
                if sampled_gt_boxes[k, 0] > max_num:
                    max_num = sampled_gt_boxes[k, 0].copy()

                    exchange = sampled_gt_boxes[j, :].copy()
                    sampled_gt_boxes[j, :] = sampled_gt_boxes[k, :].copy()
                    sampled_gt_boxes[k, :] = exchange.copy()

                    exchange = total_valid_sampled_dict[j].copy()
                    total_valid_sampled_dict[j] = total_valid_sampled_dict[k].copy(
                    )
                    total_valid_sampled_dict[k] = exchange.copy()

        if total_valid_sampled_dict.__len__() > 0:

            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        left_img = data_dict['left_img']
        right_img = data_dict['right_img']

        # brightness  augmentation
        left_img, right_img = self.trans(left_img, right_img)

        _,_, W = left_img.shape

        right_pad = 1248 - W

   

        h_shift = left_img.shape[1] - 224
        left_img = F.pad(left_img, (0, right_pad, 0, 0), "constant", 0)
        right_img = F.pad(right_img, (0, right_pad, 0, 0), "constant", 0)

        left_img = left_img[:, -224:, :]
        right_img = right_img[:, -224:, :]

 
 
        data_dict['h_shift'] = h_shift
 
        data_dict['left_img'] = left_img
        data_dict['right_img'] = right_img
        data_dict['snake'] = []

        data_dict.pop('gt_boxes_mask')
        return data_dict
