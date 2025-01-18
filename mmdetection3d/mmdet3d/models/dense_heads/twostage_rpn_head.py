import math
import numpy as np
import torch
import torch.nn as nn
import time
from ..roi_heads.bbox_heads.bbox2d_head import ConvPool, SAM, TorchInitConvModule, init_head
from mmcv.cnn import bias_init_with_prob, xavier_init, ConvModule
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from ..builder import DETECTORS, build_backbone, build_head
from mmcv.cnn import build_norm_layer
from ..roi_heads.bbox_heads.bbox2d_head import SAM
from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from ..builder import HEADS, build_head, build_roi_extractor


@HEADS.register_module()
class TwoStageRPNHead(BaseDenseHead):
    def __init__(self, in_channel, strides, feat_channel, num_classes, loss_center_heatmap,
                 roi_head, free_loss, class_balance=False, norm_cfg=dict(type='BN'), train_cfg=None, test_cfg=None, init_cfg=None):
        super(TwoStageRPNHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.strides = strides
        self.class_balance = class_balance
        self.roi_head = build_head(roi_head)
        self.origin_size = self.roi_head.bbox_head.roi_feat_size
        self.heatmap_head = nn.ModuleList()
        self.heatmap_stem = nn.ModuleList()
        self.register_buffer('batch_id', torch.arange(128, dtype=torch.float32))
        self.register_buffer('constant_one', torch.tensor([1, ], dtype=torch.float32))
        self.register_buffer('constant_zero', torch.tensor([0, ], dtype=torch.float32))
        for i, inp in enumerate(in_channel):
            self.heatmap_stem.append(nn.Sequential(
                # SAM(inp, 1, 5),
                # TorchInitConvModule(inp, feat_channel, 1, 1, 0, norm_cfg=norm_cfg),
                TorchInitConvModule(inp, feat_channel, 3, 1, 1, norm_cfg=norm_cfg),
                *[TorchInitConvModule(feat_channel, feat_channel, 3, 1, 1, norm_cfg=norm_cfg) for _ in
                  range(self.origin_size[0] // 2 - 1)],
            ))
            # feat_channel = inp
            self.heatmap_head.append(nn.Sequential(
                nn.Conv2d(feat_channel, self.num_classes, kernel_size=1, padding=0)
            ))
        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.free_loss = free_loss
        if free_loss:
            self.bbox_roi_extractor = build_roi_extractor(dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=1, aligned=False, sampling_ratio=1),
                # roi_layer=dict(type='RoIPool', output_size=1),
                out_channels=num_classes,
                featmap_strides=strides))
        self.train_cfg = train_cfg
        # self.train_cfg.nms_pre = self.train_cfg.nms_pre // 3 * 3
        self.test_cfg = test_cfg
        # self.test_cfg.nms_pre = self.test_cfg.nms_pre // 3 * 3
        if class_balance:
            assert self.train_cfg.nms_pre % num_classes == 0
            assert self.test_cfg.nms_pre % num_classes == 0
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize the weights."""
        '''
        for m in self.heatmap_stem.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        '''
        for m in self.heatmap_head:
            init_head(m, -4.6)
        self.roi_head.init_weights()

    def forward_train(self, x, img_metas, **kwargs):
        outs = self(x)
        losses = self.loss(*outs, **kwargs)
        collect = ('index_heatmap',)
        with torch.no_grad():
            # index_heatmap = kwargs['index_heatmap']
            # heatmap_new = []
            # for hm, inm in zip(outs[0], index_heatmap):
            #    heatmap_new.append(hm * (inm < self.num_classes))
            proposal_list = self.get_bboxes(*outs, img_metas, collect=collect, cfg=self.train_cfg, **kwargs)
        roi_losses, proposal_list = self.roi_head.forward_train(x, img_metas, proposal_list, **kwargs)
        # loss = roi_losses.pop('loss')
        losses.update(roi_losses)
        # losses['loss'] = losses['loss'] + loss
        return losses, proposal_list

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.heatmap_stem, self.heatmap_head)

    # @profile
    def forward_single(self, feat, heatmap_stem, heatmap_head):
        heatmap = heatmap_stem(feat)
        center_heatmap_preds = heatmap_head(heatmap).sigmoid()
        return center_heatmap_preds,

    @force_fp32(apply_to=('center_heatmap_preds',))
    def loss(self, center_heatmap_preds,
             center_heatmap_pos, center_heatmap_neg, **kwargs):
        outputs = dict()
        if self.free_loss:
            batch_size = center_heatmap_preds[0].shape[0]
            img_inds = self.batch_id[:batch_size]
            img_inds = img_inds.view(batch_size, 1).expand(batch_size, self.train_cfg.max_num_pre_img).reshape(-1, 1)
            bbox2d = kwargs['bbox2d_heatmap'].view(-1, 4)
            cls_heatmap_pos = kwargs['cls_heatmap_pos'].view(-1, self.num_classes)
            bbox2d_mask = kwargs['bbox2d_mask'].view(-1)
            rois = torch.cat((img_inds, bbox2d), dim=1)
            rois = rois[bbox2d_mask, :]
            cls_heatmap_pos = cls_heatmap_pos[bbox2d_mask, :].bool()
            bbox_feats = self.bbox_roi_extractor(center_heatmap_preds, rois).view(-1, self.num_classes)
            bbox_feats = bbox_feats[cls_heatmap_pos]  # + 1
            center_heatmap_preds = center_heatmap_preds + [bbox_feats, ]
            center_heatmap_pos = center_heatmap_pos + [self.constant_one.expand_as(bbox_feats), ]
            center_heatmap_neg = center_heatmap_neg + [self.constant_zero.expand_as(bbox_feats), ]
        # center_heatmap_preds = [i.sigmoid() for i in center_heatmap_preds]
        loss_center_heatmap, loss_center_heatmap_show = self.loss_center_heatmap(center_heatmap_preds,
                                                                                 center_heatmap_pos,
                                                                                 center_heatmap_neg)
        loss = loss_center_heatmap
        outputs['loss'] = loss
        for key, value in loss_center_heatmap_show.items():
            outputs[f'heatmap_{key}'] = value
        return outputs

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

    # @profile
    def get_bboxes(self, center_heatmap_preds, img_metas, collect=(), cfg=None, **kwargs):
        if cfg is None:
            cfg = self.train_cfg if self.train_cfg is not None else self.test_cfg
        # if len(collect) > 0:
        #    center_heatmap_preds, size_preds = kwargs['center_heatmap_pos'], kwargs['size_heatmap']
        if hasattr(cfg, 'add_gt'):
            add_gt = cfg.add_gt
        else:
            add_gt = False
        xy_min = kwargs['xy_min']
        xy_max = kwargs['xy_max']
        # center_heatmap_preds = [i.sigmoid() for i in center_heatmap_preds]
        if add_gt:
            center_heatmap_preds.extend(kwargs['center_heatmap_pos'])
        # print(center_heatmap_preds)
        tensor = [center_heatmap_preds, ]
        for col in collect:
            if add_gt:
                tensor.append(kwargs[col] * 2)
            else:
                tensor.append(kwargs[col])
        merge, size = self.merge_mul_level_pred(tensor)
        center_heatmap_preds = merge.pop(0)
        topk_bbox, topk_scores, topk_clses, batch_index = self.decode_heatmap(center_heatmap_preds, size, xy_min,
                                                                              xy_max,
                                                                              img_metas[0]['batch_input_shape'], cfg)
        for i in range(0, len(merge)):
            merge[i] = merge[i].gather(dim=2,
                                       index=batch_index.expand((merge[i].size(0), merge[i].size(1), cfg.nms_pre)))
            merge[i] = merge[i].view(-1)  # [b*k,]
        batch_size, K, _ = topk_bbox.shape
        img_inds = self.batch_id[:batch_size]
        img_inds = img_inds.view(batch_size, 1).expand_as(topk_scores).reshape(-1, 1)
        topk_bbox = topk_bbox.view(-1, 4)
        topk_scores = topk_scores.view(-1)
        topk_clses = topk_clses.view(-1)
        '''
        if cfg.min_score > 0:
            valid_mask = (topk_scores >= cfg.min_score)
            if not valid_mask.all():
                topk_bbox = topk_bbox[valid_mask, :]
                topk_scores = topk_scores[valid_mask]
                topk_clses = topk_clses[valid_mask]
                img_inds = img_inds[valid_mask, :]
                for i in range(len(merge)):
                    merge[i] = merge[i][valid_mask]
        if topk_bbox.numel() > 0:
            if cfg.roi_num > 0:
                topk_scores, keep = torch.topk(topk_scores, cfg.roi_num, dim=0)
                topk_bbox = topk_bbox[keep, :]
                topk_clses = topk_clses[keep]
                img_inds = img_inds[keep, :]
                for i in range(len(merge)):
                    merge[i] = merge[i][keep]
        '''
        topk_rois = torch.cat((img_inds, topk_bbox), dim=1)
        # erge_debug = [i.detach().cpu().numpy() for i in merge]
        # topk_scores_debug = topk_scores.detach().cpu().numpy()
        return [topk_rois, topk_scores, topk_clses, *merge]

    # @staticmethod
    def decode_heatmap(self, center_heatmap_pred, feature_size, xy_min, xy_max, batch_input_shape, cfg):
        K = cfg.nms_pre
        batch_size, class_num, hw_sum = center_heatmap_pred.shape
        inp_h, inp_w = batch_input_shape
        # print(center_heatmap_pred.shape)
        if self.class_balance:
            topk_scores, topk_inds = torch.topk(center_heatmap_pred.view(batch_size * class_num, -1), K // class_num)
            topk_scores = topk_scores.view(batch_size, -1)
            topk_clses = self.batch_id[:class_num].long()
            topk_clses = topk_clses.view(1, class_num, 1).expand((batch_size, class_num, K // class_num)).reshape(
                batch_size, -1)
            topk_inds = topk_inds.view(batch_size, -1)
        else:
            topk_scores, topk_inds = torch.topk(center_heatmap_pred.view(batch_size, -1), K)
            topk_clses = torch.div(topk_inds, hw_sum, rounding_mode='trunc')
            topk_inds = topk_inds % hw_sum
        '''
        topk_scores, topk_inds = torch.topk(center_heatmap_pred[:, 2, :], K)
        topk_clses = torch.ones_like(topk_inds, dtype=torch.long) * 2
        '''
        batch_index = topk_inds.unsqueeze(1)  # [B,1,K]
        height = 0
        width = 0
        topk_inds3 = 0
        batch_index2 = topk_inds.unsqueeze(2)  # [B,K,1]
        hw_min = 0
        hw_max = 0
        for siz in feature_size:
            h, w = siz
            hw_max = hw_max + h * w
            lev = (batch_index2 < hw_max) * (batch_index2 >= hw_min)
            height = height + lev * h
            width = width + lev * w
            topk_inds3 = topk_inds3 + lev * (batch_index2 - hw_min)
            hw_min = hw_max
        down_factor = inp_h / height
        topk_ys = (torch.div(topk_inds3, width, rounding_mode='trunc')).float()
        topk_xs = (topk_inds3 % width).float()
        topk_xy = torch.cat([topk_xs, topk_ys], dim=2)
        xy = topk_xy
        WH = (self.origin_size[0]) * 0.5
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

    # @profile
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
        # for k, v in kwargs.items():
        #    kwargs[k] = v[0]
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas, collect=(), cfg=self.test_cfg, **kwargs)
        proposal_list = self.roi_head.simple_test(x, proposal_list, img_metas, **kwargs)
        return proposal_list
