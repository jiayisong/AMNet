import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
from mmcv.cnn import bias_init_with_prob, xavier_init, build_conv_layer, build_norm_layer, ConvModule
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from mmcv.ops import batched_nms, box_iou_rotated
from mmdet3d.ops.iou3d.iou3d_utils import nms_3d_gpu, nms_bev_gpu, boxes_iou_3d_gpu
from mmdet3d.core import xywhr2xyxyr
from torch.nn.parameter import Parameter
from torch.nn import init
from .bbox2d_head import ConvPool, SAM, TorchInitConvModule, ACH3, init_head


class BasicBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 norm_cfg,
                 conv_cfg,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1], )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = out + identity
        return out


class DLA_head(nn.Module):
    def __init__(self, in_channel, level_num, norm_cfg, conv_cfg):
        super(DLA_head, self).__init__()
        self.block = nn.ModuleList([])
        self.root = nn.ModuleList([])
        self.root_num = []
        for i in range(2 ** level_num):
            self.block.append(BasicBlock(in_channel, norm_cfg, conv_cfg))
            factor_num = self.compute_2_factor_num(i + 1)
            if factor_num > 0:
                self.root.append(nn.Sequential(
                    build_conv_layer(conv_cfg, in_channel * (factor_num + 1), in_channel, 1, stride=1, padding=0,
                                     bias=False),
                    build_norm_layer(norm_cfg, in_channel)[1],
                    nn.ReLU(inplace=True),
                ))
                self.root_num.append(factor_num + 1)
            else:
                self.root_num.append(0)
                self.root.append(nn.Sequential())

    def compute_2_factor_num(self, num):
        i = 0
        while num % 2 == 0:
            i += 1
            num = num // 2
        return i

    def forward(self, x):
        root = []
        for i, b in enumerate(self.block):
            x = b(x)
            root.append(x)
            factor_num = self.compute_2_factor_num(i + 1)
            if factor_num > 0:
                r = torch.cat(root[-self.root_num[i]:], 1)
                x = self.root[i](r)
                root = root[:-self.root_num[i]]
                root.append(x)
        return x


@HEADS.register_module()
class BBox3DHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 roi_feat_size, in_channels, feat_channel, num_classes, bbox_coder,
                 loss_cls, loss_lhw, loss_d, loss_uv, loss_alpha,
                 loss_alpha_4bin, loss_xyz, loss_corner, loss_iou3d, pred, couple, norm_cfg,
                 test_cfg=None, use_cls=None, train_cfg=None, init_cfg=None):
        super(BBox3DHead, self).__init__(init_cfg)
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.feat_channel = feat_channel
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_lhw = build_loss(loss_lhw)
        self.loss_d = build_loss(loss_d)
        self.loss_uv = build_loss(loss_uv)
        self.loss_alpha = build_loss(loss_alpha)
        self.loss_alpha_4bin = build_loss(loss_alpha_4bin)
        self.loss_xyz = build_loss(loss_xyz)
        self.loss_corner = build_loss(loss_corner)
        self.loss_iou3d = build_loss(loss_iou3d)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.fp16_enabled = False
        assert couple in ['adaptive', 'couple', 'decouple']
        self.couple = couple
        self.pred_d = 'd' in pred
        self.pred_uv = 'uv' in pred
        self.pred_corner_2d = 'corner_2d' in pred
        self.pred_xyz = 'xyz' in pred
        self.pred_corner = 'corner' in pred
        self.pred_lhw = 'lhw' in pred
        self.pred_alpha = 'alpha' in pred
        self.pred_iou3d = ('iou3d' in pred) or ('iou3d_nuscene' in pred)
        self.pred_iou3d_nuscene = 'iou3d_nuscene' in pred
        self.pred_iou2d = 'iou2d' in pred
        if 'union_corner' in pred:
            self.pred_d = True
            self.pred_uv = True
            self.pred_lhw = True
            self.pred_alpha = True
            self.pred_corner = True
            self.pred_union_corner = True
            self.pred_union_center = False
        elif 'union_center' in pred:
            self.pred_d = True
            self.pred_uv = True
            self.pred_lhw = True
            self.pred_alpha = True
            self.pred_xyz = True
            self.pred_union_corner = False
            self.pred_union_center = True
        else:
            self.pred_union_corner = False
            self.pred_union_center = False
        self.use_cls = use_cls
        if use_cls == 'conv_head':
            group = 4
            assert ('4bin' in self.bbox_coder.alpha_type)
            self.d_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 1),
            ) for _ in range(num_classes)])
            self.uv_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 18 if self.pred_corner_2d else 2),
            ) for _ in range(num_classes)])
            self.lhw_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 3),
            ) for _ in range(num_classes)])
            if self.bbox_coder.alpha_type == '4bin' or self.bbox_coder.alpha_type == 'sincosv2':
                self.alpha_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 4),
                ) for _ in range(num_classes)])
            elif self.bbox_coder.alpha_type == 'my4bin':
                self.alpha_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 1),
                ) for _ in range(num_classes)])
            else:
                self.alpha_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 2),
                ) for _ in range(num_classes)])
            if '4bin' in self.bbox_coder.alpha_type:
                self.alpha_4bin_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 4),
                ) for _ in range(num_classes)])
                group += 1
            if self.pred_iou3d:
                self.iou3d_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 1),
                ) for _ in range(num_classes)])
                group += 1
            group = group * num_classes

        elif use_cls == 'conv_last':
            assert ('4bin' in self.bbox_coder.alpha_type)
            group = 4
            self.d_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 1),
            ) for _ in range(num_classes)])
            self.uv_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 18 if self.pred_corner_2d else 2),
            ) for _ in range(num_classes)])
            self.lhw_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 3),
            ) for _ in range(num_classes)])
            if self.bbox_coder.alpha_type == '4bin':
                self.alpha_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 4),
                ) for _ in range(num_classes)])
            elif self.bbox_coder.alpha_type == 'my4bin':
                self.alpha_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 1),
                ) for _ in range(num_classes)])
            else:
                self.alpha_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 2),
                ) for _ in range(num_classes)])
            if '4bin' in self.bbox_coder.alpha_type:
                self.alpha_4bin_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 4),
                ) for _ in range(num_classes)])
                group += 1
            if self.pred_iou3d:
                self.iou3d_head = nn.ModuleList([nn.Sequential(
                    nn.Linear(feat_channel, 1),
                ) for _ in range(num_classes)])
                group += 1
            group = group

        else:
            group = 4
            self.d_head = nn.Sequential(
                nn.Linear(feat_channel, 1),
            )
            self.uv_head = nn.Sequential(
                nn.Linear(feat_channel, 18 if self.pred_corner_2d else 2),
            )
            self.lhw_head = nn.Sequential(
                nn.Linear(feat_channel, 3),
            )
            if self.bbox_coder.alpha_type == '4bin' or self.bbox_coder.alpha_type == 'sincosv2' or self.bbox_coder.alpha_type == 'sincosv3':
                self.alpha_head = nn.Sequential(
                    nn.Linear(feat_channel, 4),
                )
            elif self.bbox_coder.alpha_type == 'my4bin':
                self.alpha_head = nn.Sequential(
                    nn.Linear(feat_channel, 1),
                )
            else:
                self.alpha_head = nn.Sequential(
                    nn.Linear(feat_channel, 2),
                )
            if '4bin' in self.bbox_coder.alpha_type:
                self.alpha_4bin_head = nn.Sequential(
                    nn.Linear(feat_channel, 4),
                )
                group += 1
            if self.pred_iou3d:
                self.iou3d_head = nn.Sequential(
                    nn.Linear(feat_channel, 1),
                )
                group += 1
            group = group
        if couple == 'couple':
            assert use_cls != 'conv_head'
            group = 1
        self.stem = nn.Sequential(
            ConvPool(in_channels, feat_channel * group, norm_cfg, roi_feat_size, group),
            ACH3(feat_channel * group + num_classes if use_cls == 'ach' else feat_channel * group,
                 feat_channel * group) if couple == 'adaptive' else nn.Sequential(),
            # ACH3(feat_channel * group),
            # ACH3(feat_channel * group),
        )
        self.group = group

    def init_weights(self):
        """Initialize the weights.
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        for m in self.bbox3d_stem.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        """

        if hasattr(self, 'alpha_4bin_head'):
            init_head(self.alpha_4bin_head, -4.6)
        init_head(self.d_head, 0)
        init_head(self.uv_head, 0)
        init_head(self.lhw_head, 0)
        if self.bbox_coder.alpha_type == 'sincosv3':
            init_head(self.alpha_head, -4.6)
        elif self.bbox_coder.alpha_type == 'sincosv4':
            init_head(self.alpha_head, 0)
        else:
            init_head(self.alpha_head, 0)
        if self.pred_iou3d:
            init_head(self.iou3d_head, -4.6)

    # @auto_fp16()
    # @profile
    def forward(self, x, cls=None):
        # print(x.is_contiguous())

        if self.use_cls == 'conv_head':
            x = self.stem(x)
            x = torch.chunk(x, self.group, 1)
            x = list(x)
            B, _ = x[0].shape
            g = self.group // self.num_classes
            x = [x[i * g:i * g + g] for i in range(self.num_classes)]
            d_pred = []
            uv_pred = []
            lhw_pred = []
            alpha_pred = []
            ori_pred = []
            iou3d_pred = []
            cls = F.one_hot(cls, self.num_classes).unsqueeze(1).float()
            for i, dh, uh, lh, ah, abh, ih in zip(x, self.d_head, self.uv_head, self.lhw_head, self.alpha_head,
                                                  self.alpha_4bin_head, self.iou3d_head):
                d_pred.append(dh(i[0]))
                uv_pred.append(uh(i[1]))
                lhw_pred.append(lh(i[2]))
                alpha_pred.append(ah(i[3]))
                ori_pred.append(abh(i[4]).sigmoid())
                iou3d_pred.append(ih(i[5]).sigmoid())
            d_pred = torch.stack(d_pred, 1)
            uv_pred = torch.stack(uv_pred, 1)
            lhw_pred = torch.stack(lhw_pred, 1)
            alpha_pred = torch.stack(alpha_pred, 1)
            ori_pred = torch.stack(ori_pred, 1)
            iou3d_pred = torch.stack(iou3d_pred, 1)
            d_pred = torch.bmm(cls, d_pred).view(B, -1)
            uv_pred = torch.bmm(cls, uv_pred).view(B, -1)
            lhw_pred = torch.bmm(cls, lhw_pred).view(B, -1)
            ori_pred = torch.bmm(cls, ori_pred).view(B, -1)
            iou3d_pred = torch.bmm(cls, iou3d_pred).view(B, -1)
            alpha_pred = torch.bmm(cls, alpha_pred).view(B, -1)
        elif self.use_cls == 'conv_last':
            x = self.stem(x)
            x = torch.chunk(x, self.group, 1)
            x = list(x)
            B, _ = x[0].shape
            g = self.group // self.num_classes
            d_pred = []
            uv_pred = []
            lhw_pred = []
            alpha_pred = []
            ori_pred = []
            iou3d_pred = []
            cls = F.one_hot(cls, self.num_classes).unsqueeze(1).float()
            for dh, uh, lh, ah, abh, ih in zip(self.d_head, self.uv_head, self.lhw_head, self.alpha_head,
                                               self.alpha_4bin_head, self.iou3d_head):
                if self.couple == 'couple':
                    d_pred.append(dh(x[0]))
                    uv_pred.append(uh(x[0]))
                    lhw_pred.append(lh(x[0]))
                    alpha_pred.append(ah(x[0]))
                    ori_pred.append(abh(x[0]).sigmoid())
                    iou3d_pred.append(ih(x[0]).sigmoid())
                else:
                    d_pred.append(dh(x[0]))
                    uv_pred.append(uh(x[1]))
                    lhw_pred.append(lh(x[2]))
                    alpha_pred.append(ah(x[3]))
                    ori_pred.append(abh(x[4]).sigmoid())
                    iou3d_pred.append(ih(x[5]).sigmoid())
            d_pred = torch.stack(d_pred, 1)
            uv_pred = torch.stack(uv_pred, 1)
            lhw_pred = torch.stack(lhw_pred, 1)
            alpha_pred = torch.stack(alpha_pred, 1)
            ori_pred = torch.stack(ori_pred, 1)
            iou3d_pred = torch.stack(iou3d_pred, 1)
            d_pred = torch.bmm(cls, d_pred).view(B, -1)
            uv_pred = torch.bmm(cls, uv_pred).view(B, -1)
            lhw_pred = torch.bmm(cls, lhw_pred).view(B, -1)
            ori_pred = torch.bmm(cls, ori_pred).view(B, -1)
            iou3d_pred = torch.bmm(cls, iou3d_pred).view(B, -1)
            alpha_pred = torch.bmm(cls, alpha_pred).view(B, -1)
        else:
            if self.use_cls == 'ach':
                x = self.stem[0](x)
                x = self.stem[1](x, F.one_hot(cls, self.num_classes).float() * 2 / 3 - 1 / 3)
            else:
                x = self.stem(x)
            x = torch.chunk(x, self.group, 1)
            if self.couple == 'couple':
                x = [x[0], ] * 6
            d_pred = self.d_head(x[0])
            # d_pred = torch.exp(-d_pred)
            uv_pred = self.uv_head(x[1])
            lhw_pred = self.lhw_head(x[2])
            alpha_pred = self.alpha_head(x[3])
            if self.bbox_coder.alpha_type == 'sincos':
                alpha_pred = F.normalize(alpha_pred)
            elif self.bbox_coder.alpha_type == 'sincosv2':
                alpha_pred = torch.sigmoid(alpha_pred)
                alpha_pred = torch.atan2(alpha_pred[:, 0] - alpha_pred[:, 2],
                                         alpha_pred[:, 1] - alpha_pred[:, 3]).unsqueeze(1)
            elif self.bbox_coder.alpha_type == 'sincosv3':
                alpha_pred = torch.sigmoid(alpha_pred)
            elif self.bbox_coder.alpha_type == 'sincosv4':
                alpha_pred = torch.sigmoid(alpha_pred)
                # alpha_pred = F.normalize(alpha_pred)
            elif self.bbox_coder.alpha_type == 'sincosv5':
                pass
            elif self.bbox_coder.alpha_type == 'sincosv6':
                pass
            if '4bin' in self.bbox_coder.alpha_type:
                ori_pred = self.alpha_4bin_head(x[4]).sigmoid()
            else:
                ori_pred = None
                # ori_pred = ori_pred.sigmoid()
            # if d_score is not None:
            # d_score = d_score.sigmoid()
            # d_score = d_score.sigmoid()
            if self.pred_iou3d:
                iou3d_pred = self.iou3d_head(x[5])
                iou3d_pred = iou3d_pred.sigmoid()
            else:
                iou3d_pred = None
        # print(iou3d_pred)

        return lhw_pred, uv_pred, iou3d_pred, d_pred, alpha_pred, ori_pred

    def get_targets(self, index, batch_index, **kwargs):
        index = index.long()
        batch_index = batch_index.long()
        b, n, c = kwargs['cls_heatmap_pos'].shape
        cls_heatmap_pos = kwargs['cls_heatmap_pos'].view(b * n, c)[index, :]
        cls_heatmap_neg = kwargs['cls_heatmap_neg'].view(b * n, c)[index, :]
        bbox2d_heatmap = kwargs['bbox2d_heatmap'].view(b * n, 4)[index, :]
        bbox3d_heatmap = kwargs['bbox3d_heatmap'].view(b * n, 7)[index, :]
        lhw_heatmap = kwargs['lhw_heatmap'].view(b * n, 3)[index, :]
        uv_heatmap = kwargs['uv_heatmap'].view(b * n, 2)[index, :]
        if self.pred_corner_2d:
            corners_2d_heatmap = kwargs['corners_2d_heatmap'].view(b * n, 16)[index, :]
            uv_heatmap = torch.cat((uv_heatmap, corners_2d_heatmap), dim=1)
        d_heatmap = kwargs['d_heatmap'].view(b * n, 1)[index, :]
        corner_heatmap = kwargs['corner_heatmap'].view(b * n, 3, 8)[index, :, :]
        alpha_heatmap = kwargs['alpha_heatmap'].view(b * n, -1)[index, :]
        alpha4bin_heatmap = kwargs['alpha_4bin_heatmap'].view(b * n, -1)[index, :]
        cam2img = kwargs['cam2img'][batch_index, :, :]
        img2cam = kwargs['img2cam'][batch_index, :, :]
        K_out = kwargs['K_out'][batch_index, :]
        return cls_heatmap_pos, cls_heatmap_neg, bbox2d_heatmap, bbox3d_heatmap, lhw_heatmap, uv_heatmap, d_heatmap, \
               corner_heatmap, alpha_heatmap, alpha4bin_heatmap, cam2img, img2cam, K_out

    @force_fp32(
        apply_to=('lhw_pred', 'uv_pred', 'iou3d_pred', 'd_pred', 'alpha_pred', 'ori_pred'))
    # @profile
    def loss(self, lhw_pred, uv_pred, iou3d_pred, d_pred, alpha_pred, ori_pred,
             rois, label, mask, index, **kwargs):
        cls_heatmap_pos, cls_heatmap_neg, bbox2d_heatmap, bbox3d_heatmap, lhw_heatmap, uv_heatmap, d_heatmap, \
        corner_heatmap, alpha_heatmap, alpha4bin_heatmap, cam2img, img2cam, K_out \
            = self.get_targets(index, rois[:, 0], **kwargs)

        # score_debug = score.detach().cpu().numpy()
        loss = 0
        outputs = dict()

        if self.pred_d:
            with torch.no_grad():
                d_heatmap = self.bbox_coder.encode_d(rois, d_heatmap, self.bbox_coder.decode_lhw(lhw_pred, label),
                                                     cam2img)
        if self.pred_uv:
            # uv_heatmap_debug1 = uv_heatmap.detach().cpu().numpy()
            # rois_debug = rois.detach().cpu().numpy()
            with torch.no_grad():
                uv_heatmap = self.bbox_coder.encode_uv(rois, uv_heatmap)
        # lhw_pred, uv_pred, d_pred, ori_pred, alpha_pred = lhw_heatmap, uv_heatmap, d_heatmap, alpha4bin_heatmap, alpha_heatmap
        if self.pred_xyz:
            lhw = self.bbox_coder.decode_lhw(lhw_pred, label)
            uv = self.bbox_coder.decode_uv(rois, uv_pred)
            d = self.bbox_coder.decode_d(rois, d_pred, lhw, cam2img)
            xyz = self.bbox_coder.decode_xyz(uv, d, img2cam)

        if self.pred_corner:
            if not self.pred_xyz:
                # lhw_pred, uv_pred, d_pred, ori_pred, alpha_pred = lhw_heatmap, uv_heatmap, d_heatmap, alpha4bin_heatmap, alpha_heatmap
                lhw = self.bbox_coder.decode_lhw(lhw_pred, label)
                uv = self.bbox_coder.decode_uv(rois, uv_pred)
                d = self.bbox_coder.decode_d(rois, d_pred, lhw, cam2img)
                xyz = self.bbox_coder.decode_xyz(uv, d, img2cam)
            ry = self.bbox_coder.decode_ry(ori_pred, alpha_pred, xyz, K_out)
            corner_preds = self.bbox_coder.decode_corner(xyz, lhw, ry)

        with torch.no_grad():
            bbox2d = rois[:, 1:]
            iou_heatmap2d = bbox_overlaps(bbox2d.unsqueeze(0), bbox2d_heatmap.unsqueeze(0), mode='iou',
                                          is_aligned=True).transpose(0, 1)
            mask_sum = mask.sum()
            outputs['s1s2'] = mask_sum
            if self.pred_iou3d or self.pred_union_corner or self.pred_union_center:

                if self.pred_iou3d_nuscene:
                    if (not self.pred_corner) and (not self.pred_xyz):
                        lhw = self.bbox_coder.decode_lhw(lhw_pred, label)
                        uv = self.bbox_coder.decode_uv(rois, uv_pred)
                        d = self.bbox_coder.decode_d(rois, d_pred, lhw, cam2img)
                        xyz = self.bbox_coder.decode_xyz(uv, d, img2cam)
                    iou_heatmap3d = torch.norm((xyz - bbox3d_heatmap[:, :3]), dim=1, keepdim=True)
                    iou_heatmap3d = (2 - torch.log2(iou_heatmap3d)) / 3
                    iou_heatmap3d = torch.clamp(iou_heatmap3d, 0, 1)
                    outputs['iou3d_nuscene'] = (iou_heatmap3d * mask).sum() / mask_sum
                else:
                    if not self.pred_corner:
                        if not self.pred_xyz:
                            lhw = self.bbox_coder.decode_lhw(lhw_pred, label)
                            uv = self.bbox_coder.decode_uv(rois, uv_pred)
                            d = self.bbox_coder.decode_d(rois, d_pred, lhw, cam2img)
                            xyz = self.bbox_coder.decode_xyz(uv, d, img2cam)
                        ry = self.bbox_coder.decode_ry(ori_pred, alpha_pred, xyz, K_out)
                    bbox3d = torch.cat((xyz, lhw, ry), dim=1)
                    # t = time.time()
                    iou_heatmap3d = boxes_iou_3d_gpu(bbox3d, bbox3d_heatmap, aligned=True).unsqueeze(1)
                    # print('3diou', time.time() - t)
                    # debug_iou3d = (iou_heatmap3d).detach().cpu().numpy()
                    # debug_bbox3d = bbox3d.detach().cpu().numpy()
                    # debug_bbox3d_heatmap = bbox3d_heatmap.detach().cpu().numpy()
                    outputs['iou3d'] = (iou_heatmap3d * mask).sum() / mask_sum

        if self.pred_union_corner or self.pred_union_center:
            mask_l1 = mask * (1 - iou_heatmap3d ** 0.1)
            mask_corner = mask * (iou_heatmap3d ** 0.1)
        else:
            mask_l1 = mask
            mask_corner = mask
        '''
        loss_cls, loss_cls_show = self.loss_cls(cls_pred, cls_heatmap_pos, cls_heatmap_neg)
        loss = loss_cls
        if torch.any(torch.isnan(loss)):
            raise RuntimeError(f'loss_cls_heatmap loss is nan')
        for key, value in loss_cls_show.items():
            outputs[f'cls_{key}'] = value
        '''
        # loss = loss + loss_face_heatmap
        # loss = loss + loss_h
        # loss = loss + loss_size
        # loss = loss + loss_sincos
        if self.pred_lhw:
            loss_lhw, loss_lhw_show = self.loss_lhw(lhw_pred, lhw_heatmap, mask_l1)
            loss = loss + loss_lhw
            '''
            if torch.any(torch.isnan(loss_lhw)):
                raise RuntimeError('lhw loss is nan')
            '''
            # print((lhw_heatmap * (mask > 0.991)).min(), (lhw_heatmap * (mask > 0.991)).max(), (lhw_heatmap * (mask > 0.991)).sum() / (mask > 0.991).sum())

            for key, value in loss_lhw_show.items():
                outputs[f'lhw_{key}'] = value
        if hasattr(self, 'alpha_4bin_head'):
            loss_alpha_4bin, loss_alpha_4bin_show = self.loss_alpha_4bin(ori_pred, alpha4bin_heatmap * mask,
                                                                         (1 - alpha4bin_heatmap) * mask)
            loss = loss + loss_alpha_4bin
            for key, value in loss_alpha_4bin_show.items():
                outputs[f'a4bin_{key}'] = value

        if self.pred_alpha:
            if self.bbox_coder.alpha_type == '4bin':
                loss_alpha, loss_alpha_show = self.loss_alpha(alpha_pred, alpha_heatmap, mask_l1 * alpha4bin_heatmap)
            elif self.bbox_coder.alpha_type == 'my4bin':
                loss_alpha, loss_alpha_show = self.loss_alpha(alpha_pred, alpha_heatmap, mask_l1)
            elif self.bbox_coder.alpha_type == 'sincosv2':
                a = (alpha_pred - alpha_heatmap)
                a = torch.where(a > np.pi, a - 2 * np.pi, a)
                a = torch.where(a < -np.pi, a + 2 * np.pi, a)
                loss_alpha, loss_alpha_show = self.loss_alpha(a, 0.0, mask_l1)
            elif self.bbox_coder.alpha_type == 'sincosv6':
                alpha_heatmap, theta = torch.split(alpha_heatmap, (2, 4), 1)
                theta = theta.view(-1, 2, 2)
                alpha_pred = torch.bmm(theta, alpha_pred.unsqueeze(-1))
                alpha_pred = alpha_pred.sigmoid()
                loss_alpha, loss_alpha_show = self.loss_alpha(alpha_pred.squeeze(-1), alpha_heatmap, mask_l1)
            else:
                loss_alpha, loss_alpha_show = self.loss_alpha(alpha_pred, alpha_heatmap, mask_l1)

            loss = loss + loss_alpha
            for key, value in loss_alpha_show.items():
                outputs[f'alpha_{key}'] = value
        '''
        if torch.any(torch.isnan(loss_alpha)):
            raise RuntimeError('sincos loss is nan')
        '''
        # print((alpha_heatmap * (mask > 0.991)).min(), (alpha_heatmap * (mask > 0.991)).max(), (alpha_heatmap * (mask > 0.991)).sum() / (mask > 0.991).sum())
        if self.pred_xyz:
            loss_xyz, loss_xyz_show = self.loss_xyz(xyz, bbox3d_heatmap[:, :3], mask_corner)
            loss = loss + loss_xyz
            for key, value in loss_xyz_show.items():
                outputs[f'xyz_{key}'] = value
        if self.pred_corner:
            loss_corner, loss_corner_show = self.loss_corner(corner_preds, corner_heatmap, mask_corner.unsqueeze(1))
            loss = loss + loss_corner
            '''
            if torch.any(torch.isnan(loss_corner)):
                raise RuntimeError('corner loss is nan')
            '''
            for key, value in loss_corner_show.items():
                outputs[f'corner_{key}'] = value
        if self.pred_d:
            #print('d',(d_heatmap * (mask > 0.991)).min(), (d_heatmap * (mask > 0.991)).max(), (d_heatmap * (mask > 0.991)).sum() / (mask > 0.991).sum())
            loss_d, loss_d_show = self.loss_d(d_pred, d_heatmap, mask_l1)
            loss = loss + loss_d
            # d_heatmap_debug = d_heatmap[mask > 0.99].cpu().numpy()
            '''
            if torch.any(torch.isnan(loss_d)):
                raise RuntimeError('d loss is nan')
            '''
            for key, value in loss_d_show.items():
                outputs[f'd_{key}'] = value
        if self.pred_uv:
            # uv_heatmap_debug1 = uv_heatmap.detach().cpu().numpy()
            # rois_debug = rois.detach().cpu().numpy()
            # uv_heatmap_debug = uv_heatmap.detach().cpu().numpy()
            # uv_pred_debug = uv_pred.detach().cpu().numpy()
            # mask_debug = mask.detach().cpu().numpy()
            # score_debug = score.detach().cpu().numpy()
            # index_debug = index.detach().cpu().numpy()
            # print('uv', (uv_heatmap * (mask > 0.991)).min(), (uv_heatmap * (mask > 0.991)).max(),  (uv_heatmap * (mask > 0.991)).sum() / (mask > 0.991).sum())
            loss_uv, loss_uv_show = self.loss_uv(uv_pred, uv_heatmap, mask_l1)
            loss = loss + loss_uv
            '''
            if torch.any(torch.isnan(loss_uv)):
                raise RuntimeError('uv loss is nan')
            '''
            for key, value in loss_uv_show.items():
                outputs[f'uv_{key}'] = value
        if self.pred_iou3d:
            # mask1 = mask1 * (score.unsqueeze(1) > 0.05)
            iou3d_heatmap = self.bbox_coder.encode_iou3d(iou_heatmap3d)
            loss_iou3d, loss_iou_show = self.loss_iou3d(iou3d_pred, iou3d_heatmap, mask)
            loss = loss + loss_iou3d  # / mask_sum
            '''
            if torch.any(torch.isnan(loss_iou3d)):
                raise RuntimeError('loss_iou3d is nan')
            '''
            for key, value in loss_iou_show.items():
                outputs[f'iou3d_{key}'] = value
        outputs['loss'] = loss
        # print(f'd_head_bn_conv {self.d_head.stem[0][0].norm.weight[0].item():.6f} {self.d_head.stem[0][0].conv.weight[0,0,0,0].item():.6f}')
        # for key, value in loss_face_heatmap_show.items():
        #    outputs[f'face_{key}'] = value

        # for key, value in loss_size_show.items():
        #    outputs[f'size_{key}'] = value
        # for key, value in loss_h_show.items():
        #    outputs[f'h_{key}'] = value
        s3 = torch.where(mask < 1 - 1e-3, iou_heatmap3d, iou_heatmap3d.new_ones((1,)))
        return outputs, s3

    @force_fp32(
        apply_to=('lhw_pred', 'uv_pred', 'iou3d_pred', 'd_pred', 'alpha_pred', 'ori_pred'))
    # @profile
    def get_bboxes(self, lhw_pred, uv_pred, iou3d_pred, d_pred, alpha_pred, ori_pred,
                   img_metas, rois, score, label, rescale=False, with_nms=True, **kwargs):
        img2cam = kwargs['img2cam']
        cam2img = kwargs['cam2img']
        K_out = kwargs['K_out']
        xy_max = kwargs['xy_max']
        xy_min = kwargs['xy_min']
        pad_bias = kwargs['pad_bias']
        scale_factor = kwargs['scale_factor']

        '''
        cls_heatmap_pos, cls_heatmap_neg, bbox2d_heatmap, lhw_heatmap, uv_heatmap, d_heatmap, sincos_heatmap, alpha4bin_heatmap = self.get_targets(
            index, **kwargs)
        uv_pred = self.bbox_coder.encode_uv(rois, uv_heatmap)
        d_pred = d_heatmap
        sincos_pred = sincos_heatmap
        ori_pred = alpha4bin_heatmap
        lhw_pred = lhw_heatmap
        '''
        b, _ = rois.shape
        # scores, label = self.decode_label(cls_pred)
        # score = score * scores
        label2d = label.clone()
        score2d = score.clone()
        label3d = label.clone()
        score3d = score.clone()

        if self.pred_iou3d:
            # score3d = score3d * iou3d_pred.squeeze(1)#.clamp_min(0)
            iou3d_pred = self.bbox_coder.decode_iou3d(iou3d_pred)
            score3d = score3d * (iou3d_pred.squeeze(1))  # .clamp_min(0)

        bbox2d = rois[:, 1:]
        bbox2d = (bbox2d - pad_bias) / scale_factor
        bbox2d = torch.minimum(bbox2d, xy_max)
        bbox2d = torch.maximum(bbox2d, xy_min)

        if self.test_cfg.min_bbox_size > 0 or self.test_cfg.min_score > 0:
            w = bbox2d[:, 2] - bbox2d[:, 0]
            h = bbox2d[:, 3] - bbox2d[:, 1]
            valid_mask = (w >= self.test_cfg.min_bbox_size) & (h >= self.test_cfg.min_bbox_size) & (
                    score2d >= self.test_cfg.min_score)
            bbox2d = bbox2d[valid_mask]
            score2d = score2d[valid_mask]
            label2d = label2d[valid_mask]
        if bbox2d.numel() > 0:
            if with_nms:
                bbox2d, label2d, keep1 = self.box2d_nms(bbox2d, score2d, label2d)
            if img_metas['flip']:
                bbox2d_temp = bbox2d.clone()
                bbox2d[:, 0] = xy_max[:, 0] - 1 - bbox2d_temp[:, 2]
                bbox2d[:, 2] = xy_max[:, 0] - 1 - bbox2d_temp[:, 0]
        else:
            bbox2d = bbox2d.new_zeros(0, 5)
        lhw = self.bbox_coder.decode_lhw(lhw_pred, label3d)
        uv = self.bbox_coder.decode_uv(rois, uv_pred)
        d = self.bbox_coder.decode_d(rois, d_pred, lhw, cam2img)
        xyz = self.bbox_coder.decode_xyz(uv, d, img2cam)
        ry = self.bbox_coder.decode_ry(ori_pred, alpha_pred, xyz, K_out)
        bbox3d = torch.cat((xyz, lhw, ry), dim=1)
        if self.test_cfg.min_score > 0 or self.test_cfg.max_d > 0:
            valid_mask = (score3d >= self.test_cfg.min_score) & (d.squeeze(1) <= self.test_cfg.max_d)
            bbox3d = bbox3d[valid_mask]
            score3d = score3d[valid_mask]
            label3d = label3d[valid_mask]
        if bbox3d.numel() > 0:
            if with_nms:
                bbox3d, score3d, label3d, keep2 = self.box3d_nms(bbox3d, score3d, label3d)
            if img_metas['flip']:
                bbox3d[:, 0] = -bbox3d[:, 0]
                bbox3d[:, 6] = (-bbox3d[:, 6]) % (2 * np.pi) - np.pi
        else:
            bbox3d = bbox3d.new_zeros(0, 7)
        return bbox3d, score3d, label3d, bbox2d, score2d, label2d

    def box3d_nms(self, bboxes3d, scores, labels):
        if hasattr(self.test_cfg, 'nms_3d'):
            if self.test_cfg.nms_3d.type == '3d':
                keep = nms_3d_gpu(bboxes3d, scores, self.test_cfg.nms_3d.iou_threshold,
                                  fol_maxsize=self.test_cfg.nms_3d.max_num)
                labels = labels[keep]
                scores = scores[keep]
                bboxes3d = bboxes3d[keep, :]
            elif self.test_cfg.nms_3d.type == 'bev':
                keep = nms_bev_gpu(bboxes3d, scores, self.test_cfg.nms_3d.iou_threshold,
                                   fol_maxsize=self.test_cfg.nms_3d.max_num)
                labels = labels[keep]
                scores = scores[keep]
                bboxes3d = bboxes3d[keep, :]

                '''
                x1, y1, z1, l1, h1, w1, ry1 = torch.split(bboxes3d, 1, dim=1)
                bboxes_for_nms = torch.cat((x1, z1, l1, w1, ry1), dim=1)
                bboxes_for_nms = xywhr2xyxyr(bboxes_for_nms)
                keep = nms_bev_gpu(bboxes_for_nms, scores, self.test_cfg.nms_3d.iou_threshold)
                '''
            elif self.test_cfg.nms_3d.type == 'weight_3d':
                scores_, order = scores.sort(0, descending=True)
                bboxes3d_ = bboxes3d[order].contiguous()
                N = bboxes3d.size(0)
                iou = boxes_iou_3d_gpu(bboxes3d_, bboxes3d_).unsqueeze(0).unsqueeze(0)
                # iou_debug = iou.cpu().numpy()
                iou0 = iou[0, 0, 0, :]
                mask = iou0 > self.test_cfg.nms_3d.iou_threshold
                keep = mask.new_zeros([N, ], dtype=torch.bool)
                keep[0] = True
                weights = [mask * scores_]
                for i in range(N):
                    if mask[i]:
                        continue
                    keep[i] = True
                    ioui = iou[0, 0, i, :]
                    maski = ioui > self.test_cfg.nms_3d.iou_threshold
                    maski[:i] = False
                    mask = mask + maski
                    weights.append(maski * scores_)
                weights = torch.stack(weights, 0)
                weight_sum = torch.sum(weights, dim=1, keepdim=True)
                weights = weights / weight_sum  # [M,N]
                keep = order[keep].contiguous()
                if self.test_cfg.nms_3d.max_num > 0:
                    keep = keep[:self.test_cfg.nms_3d.max_num]
                    weights = weights[:self.test_cfg.nms_3d.max_num, :]
                # weights_debug = weights.cpu().numpy()
                bboxes3d = torch.sum(bboxes3d_.unsqueeze(0) * weights.unsqueeze(2), dim=1, keepdim=False)
                labels = labels[keep]
                scores = scores[keep]
                # keep_debug = keep.cpu().numpy()
                # keep2_debug = nms_3d_gpu(bboxes3d, scores, self.test_cfg.nms_3d.iou_threshold,fol_maxsize=self.test_cfg.nms_3d.max_num).cpu().numpy()
            else:
                raise NotImplementedError

        else:
            keep = None
        return bboxes3d, scores, labels, keep

    def box2d_nms(self, bboxes, scores, labels):
        if hasattr(self.test_cfg, 'nms_2d'):
            out_bboxes, keep = batched_nms(bboxes, scores, labels, self.test_cfg.nms_2d)
            out_labels = labels[keep]
        else:
            out_bboxes = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
            keep = None
            out_labels = labels
        return out_bboxes, out_labels, keep

    def compute_iou3d(self, bboxes1, bboxes2):
        iou3d = boxes_iou_3d_gpu(bboxes1, bboxes2)
        return iou3d  # [b,1]
