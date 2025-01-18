import math

import numpy as np
import torch
import torch.nn as nn
import time
from mmcv.cnn import bias_init_with_prob, xavier_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class MyRPNHead(BaseDenseHead):
    def __init__(self, in_channel, strides, feat_channel, num_classes, iou_branch, add_gt, loss_center_heatmap,
                 loss_size, loss_iou, train_cfg=None, test_cfg=None, init_cfg=None):
        super(MyRPNHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.strides = strides
        self.iou_branch = iou_branch
        self.add_gt = add_gt
        self.heatmap_head = nn.ModuleList()
        self.heatmap_stem = nn.ModuleList()
        self.hwl_head = nn.ModuleList()
        self.size_stem = nn.ModuleList()
        self.iou_head = nn.ModuleList()
        self.size_head = nn.ModuleList()
        for i, inp in enumerate(in_channel):
            self.heatmap_stem.append(nn.Sequential(self.conv3x3(inp, feat_channel),
                                                   self.conv3x3(feat_channel, feat_channel)))
            self.heatmap_head.append(self.conv1x1(feat_channel, self.num_classes))
            self.size_stem.append(nn.Sequential(self.conv3x3(inp, feat_channel),
                                                self.conv3x3(feat_channel, feat_channel)))
            self.iou_head.append(self.conv1x1(feat_channel, 1))
            self.size_head.append(self.conv1x1(feat_channel, 4))
        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_size = build_loss(loss_size)
        self.loss_iou = build_loss(loss_iou)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def conv3x3(self, in_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        return layer

    def conv1x1(self, in_channel, out_channel):
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        for heatmap_head, size_head, iou_head in zip(self.heatmap_head, self.size_head, self.iou_head):
            heatmap_head.bias.data.fill_(bias_init_with_prob(0.01))
            size_head.bias.data[:2].fill_(2)
            size_head.bias.data[2:].fill_(0.5)
            iou_head.bias.data.fill_(bias_init_with_prob(0.01))
            # nn.init.normal_(heatmap_head.weight, std=0.01)
            # nn.init.normal_(size_head.weight, std=0.01)
            # nn.init.normal_(iou_head.weight, std=0.01)

    def forward_train(self,
                      x,
                      img_metas,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        losses, iou_heatmap = self.loss(*outs, **kwargs)
        kwargs['iou_heatmap'] = iou_heatmap
        collect = ('iou_heatmap', 'index_heatmap')
        with torch.no_grad():
            proposal_list = self.get_bboxes(*outs, img_metas, collect=collect, cfg=self.train_cfg, add_gt=self.add_gt,
                                            **kwargs)
        return losses, proposal_list

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.heatmap_stem, self.heatmap_head,
                           self.size_stem, self.iou_head, self.size_head)

    def forward_single(self, feat, heatmap_stem, heatmap_head, size_stem, iou_head, size_head):
        heatmap = heatmap_stem(feat)
        center_heatmap_preds = heatmap_head(heatmap).sigmoid()
        size_pred = size_stem(feat)
        iou_pred = iou_head(size_pred).sigmoid()
        size_pred = size_head(size_pred)
        return center_heatmap_preds, size_pred, iou_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'size_preds', 'iou_pred'))
    def loss(self, center_heatmap_preds, size_preds, iou_pred,
             center_heatmap_pos, center_heatmap_neg, size_heatmap, size_mask, **kwargs
             ):
        assert len(center_heatmap_preds) == len(size_preds) == len(center_heatmap_pos) == len(
            center_heatmap_neg) == len(size_heatmap) == len(size_mask)
        outputs = dict()
        with torch.no_grad():
            iou_heatmap = self.compute_iou(size_preds, size_heatmap)
        if self.iou_branch:
            loss_iou, loss_iou_show = self.loss_iou(iou_pred, iou_heatmap, size_mask)
            for key, value in loss_iou_show.items():
                outputs[f'iou_{key}'] = value
        else:
            loss_iou = 0
        loss_center_heatmap, loss_center_heatmap_show = self.loss_center_heatmap(center_heatmap_preds,
                                                                                 center_heatmap_pos, center_heatmap_neg)
        loss_size, loss_size_show = self.loss_size(size_preds, size_heatmap, size_mask)
        loss = loss_center_heatmap + loss_size + loss_iou
        outputs['loss'] = loss
        for key, value in loss_center_heatmap_show.items():
            outputs[f'heatmap_{key}'] = value

        for key, value in loss_size_show.items():
            outputs[f'size_{key}'] = value
        return outputs, iou_heatmap

    def compute_iou(self, pred, gt):
        iou = []
        for pre, g in zip(pred, gt):
            wh1, offset1 = torch.split(pre, 2, dim=1)
            wh2, offset2 = torch.split(g, 2, dim=1)
            wh1 = wh1.exp()
            wh2 = wh2.exp()
            wh1_half = wh1 * 0.5
            wh2_half = wh2 * 0.5
            xy_min1 = offset1 - wh1_half
            xy_min2 = offset2 - wh2_half
            xy_max1 = offset1 + wh1_half
            xy_max2 = offset2 + wh2_half
            inter = torch.minimum(xy_max1, xy_max2) - torch.maximum(xy_min1, xy_min2)
            inter = torch.prod(inter.clamp_min(0), dim=1, keepdim=True)
            area1 = torch.prod(wh1, dim=1, keepdim=True)
            area2 = torch.prod(wh2, dim=1, keepdim=True)
            iou.append(inter / (area1 + area2 - inter))
        return iou

    def merge_mul_level_pred(self, preds):
        merge = [[] for _ in range(len(preds))]
        size = []
        for i, pred in enumerate(preds):
            for level in pred:
                b, c, h, w = level.shape
                level_flatten = level.view(b, c, -1)
                merge[i].append(level_flatten)
                if i == 0:
                    size.append([h, w])
            merge[i] = torch.cat(merge[i], 2)
        return merge, size

    def get_bboxes(self, center_heatmap_preds, size_preds, iou_preds, img_metas,
                   collect=(), cfg=None, add_gt=False, **kwargs):
        if cfg is None:
            cfg = self.train_cfg if self.train_cfg is not None else self.test_cfg
        # if len(collect) > 0:
        #    center_heatmap_preds, size_preds = kwargs['center_heatmap_pos'], kwargs['size_heatmap']
        if add_gt:
            center_heatmap_preds.extend(kwargs['center_heatmap_pos'])
            size_preds.extend(kwargs['size_heatmap'])
            iou_preds = iou_preds * 2
        xy_min = kwargs['xy_min']
        xy_max = kwargs['xy_max']
        tensor = [center_heatmap_preds, size_preds, iou_preds]
        for col in collect:
            if add_gt:
                tensor.append(kwargs[col] * 2)
            else:
                tensor.append(kwargs[col])
        merge, size = self.merge_mul_level_pred(tensor)
        center_heatmap_preds, size_preds, iou_preds = merge[:3]
        if self.iou_branch:
            center_heatmap_preds = center_heatmap_preds * iou_preds
        merge[0], merge[1], merge[2], batch_index = self.decode_heatmap(center_heatmap_preds,
                                                                        size_preds, size, xy_min, xy_max,
                                                                        img_metas[0][
                                                                            'batch_input_shape'], cfg)
        for i in range(3, len(merge)):
            merge[i] = merge[i].gather(dim=2,
                                       index=batch_index.expand((merge[i].size(0), merge[i].size(1), cfg.nms_pre)))
            merge[i] = merge[i].squeeze(1)  # [b,k]
        # batch_border = batch_det_bboxes.new_tensor(border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        # batch_det_bboxes -= batch_border
        det_results = []
        # merge.append(img_metas)
        for i in zip(*merge):
            proposals, det_scores, det_labels = i[:3]
            collect = list(i[3:])
            if cfg.min_bbox_size > 0 or cfg.min_score > 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size) & (det_scores >= cfg.min_score)
                if not valid_mask.all():
                    proposals = proposals[valid_mask]
                    det_scores = det_scores[valid_mask]
                    det_labels = det_labels[valid_mask]
                    for i in range(len(collect)):
                        collect[i] = collect[i][valid_mask]
            if proposals.numel() > 0:
                if hasattr(cfg, 'nms'):
                    cfg.nms.max_num = cfg.max_per_img
                    dets, keep = batched_nms(proposals, det_scores, det_labels, cfg.nms)
                    det_labels = det_labels[keep]
                    for i in range(len(collect)):
                        collect[i] = collect[i][keep]
                else:
                    dets = torch.cat((proposals, det_scores.unsqueeze(1)), dim=1)
                    dets = dets[:cfg.max_per_img, :]
                    det_labels = det_labels[:cfg.max_per_img]
                    for i in range(len(collect)):
                        collect[i] = collect[i][:cfg.max_per_img]
            else:
                dets = proposals.new_zeros(0, 5)
            det_results.append([dets, det_labels, *collect])
        return det_results

    @staticmethod
    def decode_heatmap(center_heatmap_pred, size_pred, feature_size, xy_min, xy_max, batch_input_shape, cfg):
        K = cfg.nms_pre
        batch_size, class_num, hw_sum = center_heatmap_pred.shape
        inp_h, inp_w = batch_input_shape
        topk_scores, topk_inds = torch.topk(center_heatmap_pred.view(batch_size, -1), K)
        topk_clses = topk_inds // hw_sum
        topk_inds = topk_inds % hw_sum
        batch_index = topk_inds.unsqueeze(1)  # [B,1,K]
        # size_pred = size_pred.view(batch_size, size_pred.size(1), hw)
        size = size_pred.gather(dim=2, index=batch_index.expand((batch_size, size_pred.size(1), K)))
        size = size.transpose(1, 2)  # .exp()
        WH, offset_xy = torch.split(size, 2, dim=2)
        height = 0
        width = 0
        topk_inds3 = 0
        batch_index2 = topk_inds.unsqueeze(2)  # [B,K,1]
        hw_min = 0
        for siz in feature_size:
            h, w = siz
            hw_max = hw_min + h * w
            lev = (batch_index2 < hw_max) * (batch_index2 >= hw_min)
            height = height + lev * h
            width = width + lev * w
            topk_inds3 = topk_inds3 + lev * (batch_index2 - hw_min)
            hw_min = hw_max
        down_factor = inp_h / height
        topk_ys = (topk_inds3 // width).float()
        topk_xs = (topk_inds3 % width).float()
        topk_xy = torch.cat([topk_xs, topk_ys], dim=2)
        xy = topk_xy + offset_xy
        WH = WH.exp() * 0.5
        topk_bbox = torch.cat([xy - WH, xy + WH], dim=2) * down_factor
        # topk_bbox = torch.minimum(topk_bbox, xy_max.unsqueeze(1))
        # topk_bbox = torch.maximum(topk_bbox, xy_min.unsqueeze(1))
        return topk_bbox, topk_scores, topk_clses, batch_index

    # @staticmethod
    def _bboxes_nms(self, bboxes, scores, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels
        out_bboxes, keep = batched_nms(bboxes, scores, labels, cfg.nms)
        out_scores = out_bboxes[:, -1]
        out_bboxes = out_bboxes[:, :-1]
        out_labels = labels[keep]
        # keep_ind[keep] = keep_ind2
        return out_bboxes, out_scores, out_labels, keep

    def simple_test_rpn(self, x, img_metas, **kwargs):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        for k, v in kwargs.items():
            kwargs[k] = v[0]
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas, collect=(), cfg=self.test_cfg, **kwargs)
        return proposal_list
