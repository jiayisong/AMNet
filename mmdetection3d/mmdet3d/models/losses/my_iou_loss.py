import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, bbox_overlaps
from ..builder import LOSSES


@LOSSES.register_module()
class MyIOULoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, GIOU=False):
        super(MyIOULoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.GIOU = GIOU

    def forward(self,
                preds,
                targets,
                weights,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            preds (torch.Tensor): The prediction.
            targets (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert self.reduction in ('mean', 'sum')
        loss_bboxes = 0
        loss_num = 0
        if not isinstance(preds, list):
            preds, targets, weights = [preds, ], [targets, ], [weights, ]

        for pred, target, weight in zip(preds, targets, weights):
            loss_bbox = self.compute_iou(pred, target) * weight
            loss_bboxes = loss_bboxes + loss_bbox.sum()
            weight = weight.expand_as(loss_bbox)
            loss_num = loss_num + weight.sum()
        num = torch.clamp_min(loss_num, 1)
        if self.reduction == 'mean':
            loss_bboxes = loss_bboxes / num
            show = dict(error=loss_bboxes)
        else:
            show = dict(error=loss_bboxes / num)
        return loss_bboxes * self.loss_weight, show

    def compute_iou(self, bbox2d, bbox2d_heatmap):
        iou = bbox_overlaps(bbox2d, bbox2d_heatmap, mode='giou' if self.GIOU else 'iou', is_aligned=True).unsqueeze(1)
        return 1 - iou
