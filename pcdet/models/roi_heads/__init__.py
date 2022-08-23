from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .vpf_head import VPFHEAD
from .voxel_rcnn import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'VPFHEAD':VPFHEAD,
    'VoxelRCNNHead':VoxelRCNNHead,
 
}
