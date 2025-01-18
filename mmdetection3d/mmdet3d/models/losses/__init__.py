# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.losses import FocalLoss, SmoothL1Loss, binary_cross_entropy
from .axis_aligned_iou_loss import AxisAlignedIoULoss, axis_aligned_iou_loss
from .chamfer_distance import ChamferDistance, chamfer_distance
from .multibin_loss import MultiBinLoss
from .paconv_regularization_loss import PAConvRegularizationLoss
from .uncertain_smooth_l1_loss import UncertainL1Loss, UncertainSmoothL1Loss
from .my_l1_loss import MyL1Loss
from .my_iou_loss import MyIOULoss
from .my_gaussian_focal_loss import MyGaussianFocalLoss
from .my_cross_entropy_loss import MyCrossEntropyLoss
from .my_binary_cross_entropy_loss import MyBinaryCrossEntropyLoss
from .gaussian_varifocal_loss import GaussianVariFocalLoss

__all__ = [
    'FocalLoss', 'SmoothL1Loss', 'binary_cross_entropy', 'ChamferDistance',
    'chamfer_distance', 'axis_aligned_iou_loss', 'AxisAlignedIoULoss',
    'PAConvRegularizationLoss', 'UncertainL1Loss', 'UncertainSmoothL1Loss',
    'MultiBinLoss',
    'MyGaussianFocalLoss', 'MyCrossEntropyLoss', 'MyBinaryCrossEntropyLoss', 'MyL1Loss', 'MyIOULoss', 'GaussianVariFocalLoss'
]