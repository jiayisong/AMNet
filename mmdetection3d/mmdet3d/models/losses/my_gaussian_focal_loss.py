import mmcv
import torch.nn as nn
import torch

from ..builder import LOSSES



@mmcv.jit(derivate=True, coderize=True)
def my_gaussian_focal_loss(pred, center_heatmap_pos, center_heatmap_neg, alpha=2.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-4
    pred = torch.clamp(pred, min=eps, max=1 - eps)
    pos_loss = -pred.log() * (1 - pred).pow(alpha) * center_heatmap_pos
    neg_loss = -(1 - pred).log() * pred.pow(alpha) * center_heatmap_neg
    return pos_loss, neg_loss


@LOSSES.register_module()
class MyGaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(MyGaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.register_buffer('constant_1', torch.tensor(1, dtype=torch.float32))

    #@profile
    def forward(self, preds, center_heatmap_poses, center_heatmap_negs, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        loss = 0
        pos_losses = 0
        neg_losses = 0
        pos_nums = 0
        neg_nums = 0
        if not isinstance(preds, list):
            preds, center_heatmap_poses, center_heatmap_negs = [preds, ], [center_heatmap_poses, ], [center_heatmap_negs, ]
        for pred, center_heatmap_pos, center_heatmap_neg in zip(preds, center_heatmap_poses, center_heatmap_negs):
            pos_loss, neg_loss = my_gaussian_focal_loss(pred, center_heatmap_pos, center_heatmap_neg, alpha=self.alpha)
            pos_sum = pos_loss.sum()
            neg_sum = neg_loss.sum()
            loss = loss + pos_sum + neg_sum
            pos_losses = pos_losses + pos_sum
            neg_losses = neg_losses + neg_sum
            pos_nums = pos_nums + center_heatmap_pos.sum()
            neg_nums = neg_nums + center_heatmap_neg.sum()
        num = torch.clamp_min(pos_nums, 1)
        if self.reduction == 'mean':
            loss = loss / num
        return self.loss_weight * loss, dict(pos=pos_losses / num,
                                             neg=neg_losses / torch.clamp_min(neg_nums, 1)
                                             )
