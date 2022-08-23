import numpy as np
import torch
from ...utils import common_utils


def random_flip_along_x(gt_boxes, points, operation):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        operation[0] = 1
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    else:
        operation[0] = 0

    return gt_boxes, points, operation


def global_rotation(gt_boxes, points,  operation,  rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    operation[1] = noise_rotation
    points = common_utils.rotate_points_along_z(
        points[np.newaxis, :, :], np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(
        gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
                np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points, operation


def global_scaling(gt_boxes, points,   operation,  scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points, operation
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    operation[2] = noise_scale
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points, operation


# record the augmentation operations applied to the pasted points
def p_flip(points, operation):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = operation[0]
    if enable == 1:
        points[:, 1] = -points[:, 1]
    return points


def p_rotation(points,  operation):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = operation[1:2]
    points = rotate_points_along_z(points.unsqueeze(0), -noise_rotation)[0]

    return points


def p_scaling(points,   operation):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """

    points[:, :3] = points[:, :3] / operation[2]
    return points


def roi_flip(roi, operation):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = operation[0]
    if enable == 1:
        roi[:, 1] = -roi[:, 1]
        roi[:, 6] = - roi[:, 6]
    return roi


def roi_rotation(roi,  operation):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = operation[1:2]
    roi[:, 0:3] = rotate_points_along_z(
        roi[:, 0:3].unsqueeze(0), -noise_rotation)[0]
    roi[:, 6] -= noise_rotation
    return roi


def roi_scaling(roi,   operation):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """

    roi[:, :3] = roi[:, :3] / operation[2]
    return roi


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points, rot_matrix)
    return points_rot
