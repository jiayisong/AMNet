import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, xavier_init, build_conv_layer, build_norm_layer
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet3d.models.roi_heads.bbox_heads.bbox2d_head import ConvPool, SAM, TorchInitConvModule
from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target

from mmcv.ops import box_iou_rotated
from mmcv.runner import BaseModule
from abc import ABCMeta, abstractmethod
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet3d.core import (box3d_multiclass_nms, limit_period, points_img2cam,
                          xywhr2xyxyr)

HEAD_INIT_BOUND = 0.01


@HEADS.register_module()
class DHEDHead3d(BaseModule, metaclass=ABCMeta):
    def __init__(self, in_channel, strides, feat_channel, num_classes, loss_center_heatmap,
                 bbox_coder, pred,
                 loss_wh, loss_offset, loss_iou2d,
                 loss_lhw, loss_d, loss_uv, loss_sincos, loss_alpha,
                 loss_alpha_4bin, loss_xyz, loss_corner, loss_iou3d, loss_d_score,
                 train_cfg=None, test_cfg=None, norm_cfg=dict(type='BN'), init_cfg=None):
        super(DHEDHead3d, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.strides = strides
        self.heatmap_head = nn.ModuleList()
        self.heatmap_stem = nn.ModuleList()

        self.bbox2d_stem = nn.ModuleList()
        self.wh_head = nn.ModuleList()
        self.offset_head = nn.ModuleList()
        self.iou2d_head = nn.ModuleList()

        self.bbox3d_stem = nn.ModuleList()
        self.d_head = nn.ModuleList()
        self.uv_head = nn.ModuleList()
        self.lhw_head = nn.ModuleList()
        self.alpha_head = nn.ModuleList()
        self.alpha_4bin_head = nn.ModuleList()
        self.iou3d_head = nn.ModuleList()

        for i, inp in enumerate(in_channel):
            self.heatmap_stem.append(nn.Sequential(TorchInitConvModule(inp, feat_channel, 3, 1, 1, norm_cfg=norm_cfg),
                                                   TorchInitConvModule(feat_channel, feat_channel, 3, 1, 1,
                                                                       norm_cfg=norm_cfg), ))
            self.bbox2d_stem.append(nn.Sequential(TorchInitConvModule(inp, feat_channel, 3, 1, 1, norm_cfg=norm_cfg),
                                                  TorchInitConvModule(feat_channel, feat_channel, 3, 1, 1,
                                                                      norm_cfg=norm_cfg), ))
            self.bbox3d_stem.append(nn.Sequential())
            self.heatmap_head.append(nn.Sequential(nn.Conv2d(feat_channel, self.num_classes, kernel_size=1, padding=0)))
            self.offset_head.append(nn.Sequential(nn.Conv2d(feat_channel, 2, kernel_size=1, padding=0)))
            self.wh_head.append(nn.Sequential(nn.Conv2d(feat_channel, 2, kernel_size=1, padding=0)))
            self.iou2d_head.append(nn.Sequential(nn.Conv2d(feat_channel, 1, kernel_size=1, padding=0)))

            self.lhw_head.append(self.bbox3d_head(inp, feat_channel, 3, norm_cfg))
            self.iou3d_head.append(self.bbox3d_head(inp, feat_channel, 1, norm_cfg))
            self.uv_head.append(self.bbox3d_head(inp, feat_channel, 2, norm_cfg))
            self.d_head.append(self.bbox3d_head(inp, feat_channel, 1, norm_cfg))
            self.alpha_head.append(self.bbox3d_head(inp, feat_channel, 1, norm_cfg))
            self.alpha_4bin_head.append(self.bbox3d_head(inp, feat_channel, 4, norm_cfg))
            w = train_cfg.size[1] // strides[i]
            h = train_cfg.size[0] // strides[i]
            x = torch.arange(0, w)
            x = x.unsqueeze(0).expand([h, w]).unsqueeze(0).unsqueeze(0)
            y = torch.arange(0, h)
            y = y.unsqueeze(1).expand([h, w]).unsqueeze(0).unsqueeze(0)
            self.register_buffer(f'x{i}', x)
            self.register_buffer(f'y{i}', y)
            self.x.append(f'x{i}')
            self.y.append(f'y{i}')
        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_lhw = build_loss(loss_lhw)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_iou3d = build_loss(loss_iou3d)
        self.loss_d = build_loss(loss_d)
        self.loss_d_score = build_loss(loss_d_score)
        self.loss_offset = build_loss(loss_offset)
        self.loss_wh = build_loss(loss_wh)
        self.loss_sincos = build_loss(loss_sincos)
        self.loss_uv = build_loss(loss_uv)
        self.loss_alpha = build_loss(loss_alpha)
        self.loss_alpha_4bin = build_loss(loss_alpha_4bin)
        self.loss_xyz = build_loss(loss_xyz)
        self.loss_corner = build_loss(loss_corner)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.pred_bbox2d = 'bbox2d' in pred
        self.pred_iou = 'iou' in pred
        self.pred_d_offset = 'd' in pred
        self.pred_xyz = 'xyz' in pred
        self.pred_corner = 'corner' in pred
        self.pred_alpha = 'alpha' in pred
        self.pred_lhw = 'lhw' in pred
        self.pred_sincos = 'sincos' in pred

    def bbox3d_head(self, inp, feat_channel, out_channel, norm_cfg):
        """Build head for each branch."""
        layer = nn.Sequential(TorchInitConvModule(inp, feat_channel, 3, 1, 1, norm_cfg=norm_cfg),
                              TorchInitConvModule(feat_channel, feat_channel, 3, 1, 1, norm_cfg=norm_cfg),
                              nn.Conv2d(feat_channel, out_channel, kernel_size=1, padding=0),
                              )
        return layer

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        for heatmap_head, iou3d_head, uv_head, d_head, lhw_head, alpha_head, alpha_4bin_head, \
            offset_head, wh_head, iou2d_head in zip(
            self.heatmap_head, self.iou3d_head, self.uv_head, self.d_head, self.lhw_head,
            self.alpha_head, self.alpha_4bin_head, self.offset_head, self.wh_head, self.iou2d_head):
            heatmap_head[-1].bias.data.fill_(bias_init_with_prob(0.01))
            iou3d_head[-1].bias.data.fill_(bias_init_with_prob(0.01))
            iou2d_head[-1].bias.data.fill_(bias_init_with_prob(0.01))
            alpha_4bin_head.bias.data.fill_(bias_init_with_prob(0.01))
            nn.init.uniform_(heatmap_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(iou3d_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(uv_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(d_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(lhw_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(alpha_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(alpha_4bin_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(offset_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(wh_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            nn.init.uniform_(iou3d_head[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)

    def forward_train(self, x, img_metas, **kwargs):
        outs = self(x)
        losses = self.loss(*outs, **kwargs)
        return losses

    def simple_test(self, feats, img_metas, **kwargs):
        return self.simple_test_bboxes(feats, img_metas, **kwargs)

    def simple_test_bboxes(self, feats, img_metas, **kwargs):
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, **kwargs)
        return results_list

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.heatmap_stem, self.bbox2d_stem, self.bbox3d_stem,
                           self.heatmap_head, self.iou3d_head, self.uv_head, self.d_head, self.lhw_head,
                           self.alpha_head, self.alpha_4bin_head, self.offset_head, self.wh_head, self.iou2d_head)

    def forward_single(self, feat, heatmap_stem, bbox2d_stem, bbox3d_stem, heatmap_head, iou3d_head, uv_head,
                       d_head, lhw_head, alpha_head, alpha_4bin_head, offset_head, wh_head, iou2d_head):
        heatmap = heatmap_stem(feat)
        center_heatmap_pred = heatmap_head(heatmap).sigmoid()
        bbox2d = bbox2d_stem(feat)
        iou2d_pred = iou2d_head(bbox2d).sigmoid()
        wh_pred = wh_head(bbox2d)
        offset_pred = offset_head(bbox2d)
        bbox3d = bbox3d_stem(feat)
        lhw_pred = lhw_head(bbox3d)
        d_pred = d_head(bbox3d)
        iou3d_pred = iou3d_head(bbox3d).sigmoid()
        uv_pred = uv_head(bbox3d)
        alpha_pred = alpha_head(bbox3d)
        alpha_4bin_pred = alpha_4bin_head(bbox3d).sigmoid()
        return center_heatmap_pred, wh_pred, offset_pred, iou2d_pred, \
               lhw_pred, uv_pred, d_pred, alpha_pred, alpha_4bin_pred, iou3d_pred

    @force_fp32(apply_to=(
            'center_heatmap_preds', 'face_heatmap_preds', 'lhw_preds', 'offset_preds', 'iou_preds', 'size_preds',
            'h_preds'))
    def loss(self, center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds, iou_preds, size_preds, h_preds,
             d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_preds,
             center_heatmap_pos, center_heatmap_neg, face_heatmap, h_heatmap, d_heatmap, size_mask, size_heatmap,
             lhw_heatmap, offset_heatmap, sincos_heatmap, ltrb_heatmap, alpha_heatmap, alpha_4bin_heatmap, bbox3d_bk7,
             xyz_heatmap, corner_heatmap, img2cam, cam2img, K_out
             ):
        assert len(center_heatmap_preds) == len(lhw_preds) == len(size_preds) == len(center_heatmap_pos) == len(
            center_heatmap_neg) == len(size_heatmap) == len(size_mask)
        # face_heatmap_neg = [(1 - pos) * sm for pos, sm in zip(face_heatmap, size_mask)]
        # for lhw_pred, size_pred, x, y, s in zip(lhw_preds, size_preds, self.x, self.y, self.strides):
        outputs = dict()
        loss_center_heatmap, loss_center_heatmap_show = self.loss_center_heatmap(center_heatmap_preds,
                                                                                 center_heatmap_pos, center_heatmap_neg)
        loss = loss_center_heatmap
        if torch.any(torch.isnan(loss)):
            print('center_heatmap')
            if torch.any(torch.isnan(center_heatmap_preds[0])):
                print(center_heatmap_preds)
            raise RuntimeError('loss is nan')
        for key, value in loss_center_heatmap_show.items():
            outputs[f'center_{key}'] = value
        # loss_face_heatmap, loss_face_heatmap_show = self.loss_face_heatmap(face_heatmap_preds,face_heatmap, face_heatmap_neg)
        # loss_size, loss_size_show = self.loss_size(size_preds, size_heatmap, size_mask)
        # loss_h, loss_h_show = self.loss_h(h_preds, h_heatmap, size_mask)
        loss_alpha_4bin, loss_alpha_4bin_show = self.loss_alpha_4bin(alpha_4bin_preds, alpha_4bin_heatmap, size_mask)
        loss = loss + loss_alpha_4bin
        if torch.any(torch.isnan(loss_alpha_4bin)):
            print('alpha_4bin')
            raise RuntimeError('loss is nan')
        for key, value in loss_alpha_4bin_show.items():
            outputs[f'a4bin_{key}'] = value
        # loss = loss + loss_face_heatmap
        # loss = loss + loss_h
        # loss = loss + loss_size
        # loss = loss + loss_sincos
        if self.pred_lhw:
            loss_lhw, loss_lhw_show = self.loss_lhw(lhw_preds, lhw_heatmap, size_mask)
            loss = loss + loss_lhw
            if torch.any(torch.isnan(loss_lhw)):
                print('alpha_lhw')
                raise RuntimeError('loss is nan')
            for key, value in loss_lhw_show.items():
                outputs[f'lhw_{key}'] = value
        if self.pred_alpha:
            loss_alpha, loss_alpha_show = self.loss_alpha(alpha_preds, alpha_heatmap, size_mask)
            loss = loss + loss_alpha
            if torch.any(torch.isnan(loss_alpha)):
                print('alpha')
                raise RuntimeError('loss is nan')
            for key, value in loss_alpha_show.items():
                outputs[f'a_{key}'] = value
        if self.pred_sincos:
            loss_sincos, loss_sincos_show = self.loss_sincos(sincos_preds, sincos_heatmap, size_mask)
            loss = loss + loss_sincos
            if torch.any(torch.isnan(loss_sincos)):
                print('sincos')
                raise RuntimeError('loss is nan')
            for key, value in loss_sincos_show.items():
                outputs[f'sincos_{key}'] = value
        if self.pred_xyz:
            xyz_preds = self.decode_center(offset_preds, d_preds, d_heatmap, img2cam, cam2img)
            loss_xyz, loss_xyz_show = self.loss_xyz(xyz_preds, xyz_heatmap, size_mask)
            loss = loss + loss_xyz
            for key, value in loss_xyz_show.items():
                outputs[f'xyz_{key}'] = value
        if self.pred_corner:
            corner_preds = self.decode_corners(center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds,
                                               iou_preds, size_preds, h_preds, d_preds, sincos_preds, ltrb_preds,
                                               alpha_preds, alpha_4bin_heatmap, img2cam, cam2img, K_out)
            # corner_preds = self.decode_corners(center_heatmap_pos, face_heatmap, lhw_heatmap, offset_heatmap,
            #                                        iou_preds, size_preds, h_heatmap, d_heatmap, sincos_heatmap, ltrb_heatmap,
            #                                          alpha_heatmap, alpha_4bin_heatmap, img2cam, cam2img, K_out)
            loss_corner, loss_corner_show = self.loss_corner(corner_preds, corner_heatmap, size_mask)
            loss = loss + loss_corner
            if torch.any(torch.isnan(loss_corner)):
                print('corner')
                raise RuntimeError('loss is nan')
            for key, value in loss_corner_show.items():
                outputs[f'corner_{key}'] = value
        if self.pred_d_offset:
            loss_d, loss_d_show = self.loss_d(d_preds, d_heatmap, size_mask)
            loss_offset, loss_offset_show = self.loss_offset(offset_preds, offset_heatmap, size_mask)
            loss = loss + loss_d
            loss = loss + loss_offset
            if torch.any(torch.isnan(loss_d)):
                print('d')
                raise RuntimeError('loss is nan')
            if torch.any(torch.isnan(loss_offset)):
                print('offset')
                raise RuntimeError('loss is nan')
            for key, value in loss_offset_show.items():
                outputs[f'offset_{key}'] = value
            for key, value in loss_d_show.items():
                outputs[f'd_{key}'] = value
        if self.pred_bbox2d:
            loss_ltrb, loss_ltrb_show = self.loss_ltrb(ltrb_preds, ltrb_heatmap, size_mask)
            loss = loss + loss_ltrb
            for key, value in loss_ltrb_show.items():
                outputs[f'2d_{key}'] = value
        if self.pred_iou:
            iou_heatmap = self.compute_iou3d(center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds,
                                             iou_preds,
                                             size_preds, h_preds,
                                             d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_preds,
                                             center_heatmap_pos, center_heatmap_neg, face_heatmap, h_heatmap, d_heatmap,
                                             size_mask, size_heatmap,
                                             lhw_heatmap, offset_heatmap, sincos_heatmap, ltrb_heatmap, alpha_heatmap,
                                             alpha_4bin_heatmap, bbox3d_bk7, img2cam, cam2img, K_out)
            loss_iou, loss_iou_show = self.loss_iou(iou_preds, iou_heatmap, size_mask)
            loss = loss + loss_iou
            for key, value in loss_iou_show.items():
                outputs[f'iou_{key}'] = value

        outputs['loss'] = loss

        # for key, value in loss_face_heatmap_show.items():
        #    outputs[f'face_{key}'] = value

        # for key, value in loss_size_show.items():
        #    outputs[f'size_{key}'] = value
        # for key, value in loss_h_show.items():
        #    outputs[f'h_{key}'] = value
        return outputs

    # @profile
    def get_bboxes(self, center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds, iou_preds, size_preds,
                   h_preds, d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_preds, img_metas,
                   img2cam, cam2img, pad_bias, scale_factor, xy_max, K_out,
                   # center_heatmap_pos, center_heatmap_neg, face_heatmap, h_heatmap, d_heatmap, size_mask, size_heatmap,
                   # lhw_heatmap, offset_heatmap, sincos_heatmap, ltrb_heatmap, alpha_heatmap, alpha_4bin_heatmap,
                   ):

        # center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds, size_preds, h_preds, d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_preds = center_heatmap_pos[0], face_heatmap[0], lhw_heatmap[0], offset_heatmap[0], size_heatmap[0], h_heatmap[0], d_heatmap[0], sincos_heatmap[0], ltrb_heatmap[0], alpha_heatmap[0], alpha_4bin_heatmap[0]
        # h_preds = h_heatmap[0]
        # lhw_preds = lhw_heatmap[0]
        # offset_preds = offset_heatmap[0]
        # sincos_preds = sincos_heatmap[0]
        # center_heatmap_preds = center_heatmap_pos[0]
        # d_preds =d_heatmap[0]
        batch_bboxes2d, batch_bboxes, batch_scores, batch_labels = [], [], [], []
        for center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred, d_pred, \
            sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred, stride in zip(center_heatmap_preds, face_heatmap_preds,
                                                                               lhw_preds, offset_preds, iou_preds,
                                                                               size_preds,
                                                                               h_preds, d_preds, sincos_preds,
                                                                               ltrb_preds,
                                                                               alpha_preds, alpha_4bin_preds,
                                                                               self.strides):
            if self.pred_iou:
                center_heatmap_pred = center_heatmap_pred * iou_pred
            # alpha_4bin_pred = alpha_4bin_pred.new_zeros(ltrb_pred.size()).scatter_(1, alpha_4bin_pred, 1)
            batch_bbox2d, batch_bbox, batch_score, batch_label = self.decode_topk(
                center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred, d_pred,
                sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred, img2cam[0], cam2img[0], pad_bias[0],
                scale_factor[0], xy_max[0], K_out[0], stride)

            batch_scores.append(batch_score)
            batch_labels.append(batch_label)
            batch_bboxes.append(batch_bbox)
            batch_bboxes2d.append(batch_bbox2d)
        batch_bboxes = torch.cat(batch_bboxes, dim=1)
        batch_scores = torch.cat(batch_scores, dim=1)
        batch_labels = torch.cat(batch_labels, dim=1)
        batch_bboxes2d = torch.cat(batch_bboxes2d, dim=1)
        result_list = []
        for img_id in range(len(img_metas)):
            bboxes = batch_bboxes[img_id, :]
            bboxes2d = batch_bboxes2d[img_id, :]
            scores = batch_scores[img_id]
            labels = batch_labels[img_id, 0]
            labels2d = labels[:]
            '''
            keep_idx = scores > 0.25
            bboxes = bboxes[keep_idx]
            bboxes2d = bboxes2d[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            '''
            if labels.numel() > 0:
                bboxes2d, labels2d, keep1 = self.box2d_nms(bboxes2d, scores, labels2d)
                # bboxes = bboxes[keep1]
                # scores = scores[keep1]
                # labels = labels[keep1]
                bboxes, scores, labels, keep2 = self.box_bev_nms(bboxes, scores, labels)

            bboxes = img_metas[img_id]['box_type_3d'](bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
            attrs = None
            if self.pred_bbox2d:
                result_list.append((bboxes, scores, labels, attrs, bboxes2d, labels2d))
            else:
                result_list.append((bboxes, scores, labels, attrs))

        return result_list

    # @staticmethod
    def decode_topk(self, center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred,
                    d_pred, sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred,
                    img2cam, cam2img, pad_bias, scale_factor, xy_max, K_out, stride, topk=100):
        def gather_pos(batch_index, pred):
            a = []
            for i in pred:
                j = i.view(batch_size, i.size(1), hw)
                j = j.gather(dim=2, index=batch_index.expand((batch_size, i.size(1), topk)))
                # j = j.transpose(1, 2)
                a.append(j)
            return a

        batch_size, class_num, height, width = center_heatmap_pred.shape
        hw = height * width
        # center_heatmap_pred = (F.max_pool2d(center_heatmap_pred, 3, 1, 1) == center_heatmap_pred) * center_heatmap_pred
        topk_scores, topk_inds = torch.topk(center_heatmap_pred.view(batch_size, -1), topk)
        labels = (topk_inds // hw).unsqueeze(1)
        topk_inds = topk_inds % hw
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).float()
        batch_scores = topk_scores  # [b,k]
        batch_index = topk_inds.unsqueeze(1)
        face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred, d_pred, sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred = gather_pos(
            batch_index, (
                face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred, d_pred, sincos_pred, ltrb_pred,
                alpha_pred, alpha_4bin_pred))
        topk_xy = torch.stack([topk_xs, topk_ys], dim=1)
        uv = (topk_xy + offset_pred) * stride
        bbox2d = self.decode_bbox2d(topk_xy, ltrb_pred, stride, pad_bias, scale_factor)
        assert not (self.pred_sincos and self.pred_alpha)
        if self.pred_sincos:
            alpha_pred = sincos_pred

        if self.score_with_2d_iou:
            assert self.pred_bbox2d
            bbox3d = self.decode_bbox3d(uv, d_pred, cam2img, img2cam, K_out, lhw_pred, alpha_4bin_pred, alpha_pred,
                                        labels,
                                        tranpose=False)
            corners = self.bbox3d2corners(bbox3d)
            bbox2d_from_3d = self.corners2bbox2d(corners, cam2img, pad_bias, xy_max)
            iou = self.compute_iou(bbox2d_from_3d, bbox2d).squeeze(2)
            batch_scores = batch_scores * iou
            bbox3d = bbox3d.transpose(1, 2)
        else:
            bbox3d = self.decode_bbox3d(uv, d_pred, cam2img, img2cam, K_out, lhw_pred, alpha_4bin_pred, alpha_pred,
                                        labels)
        return bbox2d, bbox3d, batch_scores, labels

    def compute_iou(self, bbox1, bbox2):
        '''

        Args:
            bbox1: [b,k,4]
            bbox2: [b,k,4]

        Returns:
            iou: [b,k,1]
        '''
        x_min1, y_min1, x_max1, y_max1 = torch.split(bbox1, 1, dim=2)
        x_min2, y_min2, x_max2, y_max2 = torch.split(bbox2, 1, dim=2)
        area1 = (y_max1 - y_min1) * (x_max1 - x_min1)
        area2 = (y_max2 - y_min2) * (x_max2 - x_min2)
        inter_w = torch.minimum(x_max1, x_max2) - torch.maximum(x_min1, x_min2)
        inter_h = torch.minimum(y_max1, y_max2) - torch.maximum(y_min1, y_min2)
        inter = inter_h.clamp_min(0) * inter_w.clamp_min(0)
        iou = inter / (area1 + area2 - inter)
        return iou

    def decode_corners(self, center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds, iou_preds, size_preds,
                       h_preds, d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_heatmap,
                       img2cam, cam2img, K_out):
        corners = []
        for center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred, \
            d_pred, sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred, \
            stride, x, y in zip(center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds,
                                iou_preds, size_preds, h_preds,
                                d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_heatmap,
                                self.strides, self.x, self.y):
            b, _, h, w = d_pred.shape
            x = getattr(self, x)
            y = getattr(self, y)
            # alpha_4bin_pred = alpha_4bin_pred.new_zeros(ltrb_pred.size()).scatter_(1, alpha_4bin_pred, 1)

            _, bbox3d, _, _ = self.decode_all(center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred,
                                              iou_pred, size_pred, h_pred, d_pred, sincos_pred, ltrb_pred, alpha_pred,
                                              alpha_4bin_pred, img2cam, cam2img, K_out, stride, x, y, transpose=False)
            corner = self.bbox3d2corners(bbox3d)
            corners.append(corner.reshape(b, 24, h, w))
        return corners

    def bbox3d2corners(self, bbox3d):
        '''
        Args:
            bbox3d: [b,7,k]
        Returns:
            corners: [b, 8, 3, k]
        '''
        b, _, k = bbox3d.shape
        xyz, lhw, r_y = torch.split(bbox3d, (3, 3, 1), dim=1)
        cos_ry = torch.cos(r_y)
        sin_ry = torch.sin(r_y)
        zero = self.constant_0.expand(b, 1, k)
        one = self.constant_1.expand(b, 1, k)
        rot = torch.cat((cos_ry, zero, sin_ry, zero, one, zero, -sin_ry, zero, cos_ry), dim=1).view([b, 3, 3, k])
        temp2 = lhw.unsqueeze(2) * self.cor.unsqueeze(3)
        temp3 = torch.einsum("ijkm,iknm->injm", rot, temp2)
        temp4 = temp3 + xyz.unsqueeze(1)
        corners = temp4
        return corners

    def corners2bbox2d(self, corners, cam2img, pad_bias, xy_max, transpose=True):
        '''

        Args:
            corners: [b,8,3,k]
            cam2img:[b,4,4]
            pad_bias:[b,4]
            xy_max:[b,2]
        Returns:
            bbox2d:[b,k,4]
        '''
        b, _, _, k = corners.shape
        corners = torch.cat((corners, self.constant_1.unsqueeze(0).expand(b, 8, 1, k)), dim=2)
        corners[:, :, 2, :] = corners[:, :, 2, :].clamp_min(0.1)
        uvd = torch.matmul(cam2img.unsqueeze(1), corners)
        uv, d, _ = torch.split(uvd, (2, 1, 1), dim=2)
        uv = uv / d
        uv_min, _ = torch.min(uv, dim=1, keepdim=False)
        uv_min = torch.maximum(uv_min, pad_bias[:, :2].unsqueeze(2))
        uv_max, _ = torch.max(uv, dim=1, keepdim=False)
        uv_max = torch.minimum(uv_max, xy_max.unsqueeze(2))
        bbox2d = torch.cat((uv_min, uv_max), dim=1)
        if transpose:
            bbox2d = bbox2d.transpose(1, 2)
        return bbox2d

    def decode_all(self, center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred,
                   d_pred, sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred,
                   img2cam, cam2img, K_out, stride, x, y, transpose=True):
        '''

        Args:
            center_heatmap_pred:
            face_heatmap_pred:
            lhw_pred:
            offset_pred:
            iou_pred:
            size_pred:
            h_pred:
            d_pred:
            sincos_pred:
            ltrb_pred:
            alpha_pred:
            alpha_4bin_pred:
            img2cam:
            cam2img:
            stride:
            x:
            y:

        Returns:
            bbox3d:[b,k,7]
        '''
        scores, labels = torch.max(center_heatmap_pred, dim=1, keepdim=True)
        topk_xy = torch.cat([x, y], dim=1)
        uv = (topk_xy + offset_pred) * stride
        uv, d_pred, lhw_pred, alpha_4bin_pred, alpha_pred, labels, topk_xy, ltrb_pred, sincos_pred = self.flattening_pred(
            [uv, d_pred, lhw_pred, alpha_4bin_pred, alpha_pred, labels, topk_xy, ltrb_pred, sincos_pred])
        bbox2d = self.decode_bbox2d(topk_xy, ltrb_pred, stride, tranpose=transpose)
        assert not (self.pred_sincos and self.pred_alpha)
        if self.pred_sincos:
            bbox3d = self.decode_bbox3d(uv, d_pred, cam2img, img2cam, K_out, lhw_pred, alpha_4bin_pred, sincos_pred,
                                        labels,
                                        tranpose=transpose)
        else:
            bbox3d = self.decode_bbox3d(uv, d_pred, cam2img, img2cam, K_out, lhw_pred, alpha_4bin_pred, alpha_pred,
                                        labels,
                                        tranpose=transpose)
        return bbox2d, bbox3d, scores, labels

    def decode_center(self, offset_preds, d_preds, d_heatmaps, img2cam, cam2img):
        xyzs = []
        for offset_pred, d_pred, d_heatmap, stride, x, y in zip(offset_preds, d_preds, d_heatmaps, self.strides, self.x,
                                                                self.y):
            b, _, h, w = d_pred.shape
            x = getattr(self, x)
            y = getattr(self, y)
            topk_xy = torch.cat([x, y], dim=1)
            uv = (topk_xy + offset_pred) * stride
            uv, d_pred, d_heatmap = self.flattening_pred([uv, d_pred, d_heatmap])
            d_pred = self.decode_d(d_pred, cam2img)
            # d_heatmap = self.decode_d(d_heatmap, cam2img)
            xyz = self.decode_xyz(uv, d_pred, img2cam)
            xyzs.append(xyz.view(b, 3, h, w))
        return xyzs

    def compute_iou3d(self, center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds, iou_preds, size_preds,
                      h_preds, d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_preds,
                      center_heatmap_pos, center_heatmap_neg, face_heatmap, h_heatmap, d_heatmap, size_mask,
                      size_heatmap,
                      lhw_heatmap, offset_heatmap, sincos_heatmap, ltrb_heatmap, alpha_heatmap, alpha_4bin_heatmap,
                      bbox3d_bk7, img2cam, cam2img, K_out):

        # center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds, size_preds, h_preds, d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_preds =         center_heatmap_pos, face_heatmap, lhw_heatmap, offset_heatmap, size_heatmap, h_heatmap, d_heatmap, sincos_heatmap, ltrb_heatmap, alpha_heatmap, alpha_4bin_heatmap
        with torch.no_grad():
            ious = []
            for center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred, iou_pred, size_pred, h_pred, \
                d_pred, sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred, \
                stride, x, y, bboxes1 in zip(center_heatmap_preds, face_heatmap_preds, lhw_preds, offset_preds,
                                             iou_preds, size_preds, h_preds,
                                             d_preds, sincos_preds, ltrb_preds, alpha_preds, alpha_4bin_preds,
                                             self.strides, self.x, self.y,
                                             bbox3d_bk7):
                x = getattr(self, x)
                y = getattr(self, y)
                # alpha_4bin_pred = alpha_4bin_pred.new_zeros(ltrb_pred.size()).scatter_(1, alpha_4bin_pred, 1)
                _, bboxes2, _, _ = self.decode_all(center_heatmap_pred, face_heatmap_pred, lhw_pred, offset_pred,
                                                   iou_pred,
                                                   size_pred,
                                                   h_pred,
                                                   d_pred, sincos_pred, ltrb_pred, alpha_pred, alpha_4bin_pred,
                                                   img2cam, cam2img, K_out, stride, x, y)
                b, k, _ = bboxes1.shape
                x1, y1, z1, l1, h1, w1, ry1 = torch.split(bboxes1, 1, dim=2)
                x2, y2, z2, l2, h2, w2, ry2 = torch.split(bboxes2, 1, dim=2)
                bboxes1 = torch.cat((x1, z1, l1, w1, ry1), dim=2).view(-1, 5)
                bboxes2 = torch.cat((x2, z2, l2, w2, ry2), dim=2).view(-1, 5)
                # iou3d = x1 - x2#+torch.abs(z1 - z2)+torch.abs(l1 - l2)+torch.abs(w1 - w2)+torch.abs(h1 - h2)
                w1l1 = l1 * w1
                w1l1h1 = w1l1 * h1
                w2l2h2 = w2 * l2 * h2
                inter_bev = box_iou_rotated(bboxes1, bboxes2, mode='iof', aligned=True).view(b, k, 1)
                inter_bev = inter_bev * w1l1
                h1_half = h1 * 0.5
                h2_half = h2 * 0.5
                inter_h = (torch.minimum(y1 + h1_half, y2 + h2_half) - torch.maximum(y1 - h1_half,
                                                                                     y2 - h2_half)).clamp_min(0)
                inter = inter_h * inter_bev
                iou3d = inter / (w1l1h1 + w2l2h2 - inter)
                iou3d = iou3d.transpose(1, 2)  # [b,1,k]
                iou3d = iou3d.view_as(iou_pred)  # [b,1,h,w]
                ious.append(iou3d)
            return ious

    def flattening_pred(self, pre_tensor_list):
        a = []
        for i in pre_tensor_list:
            a.append(i.view(i.shape[0], i.shape[1], -1))
        return a

    def decode_bbox2d(self, topk_xy, ltrb_pred, stride, pad_bias=None, scale_factor=None, tranpose=True):
        '''

        Args:
            topk_xy: [b,2,k]
            ltrb_pred: [b,4,k]
            stride: int
            pad_bias: [b,4]
            scale_factor: [b,4]
        Returns:
            bbox2d:[b,k,4]
        '''
        # offset, wh = torch.split(ltrb_pred, 2, dim=1)
        ltrb_pred = ltrb_pred.exp() - 1
        lt, rb = torch.split(ltrb_pred, 2, dim=1)
        bbox2d = torch.cat([topk_xy - lt, topk_xy + rb], dim=1) * stride
        if pad_bias is not None:
            bbox2d = (bbox2d - pad_bias.unsqueeze(2)) / scale_factor.unsqueeze(2)
        if tranpose:
            bbox2d = bbox2d.transpose(1, 2)
        return bbox2d

    def decode_bbox3d(self, uv, d_pred, cam2img, img2cam, K_out, lhw_pred, alpha_4bin_pred, alpha_pred, labels,
                      tranpose=True):
        '''
        Args:
            uv: [b,2,k]
            d_pred: [b,1,k]
            cam2img: [b,4,4]
            img2cam:[b,4,4]
            lhw_pred:[b,3,k]
            alpha_4bin_pred: [b,4,k] or [b,1,k]
            alpha_pred: [b,1,k]
            labels:[b,k]
        Returns:
            bbox3d:[b,k,7]
        '''
        lhw_pred = self.decode_lhw(lhw_pred, labels)
        # d = cam2img[:, 1, 1].view(batch_size, 1, 1) * lhw_pred[:, 1, :].unsqueeze(1) / (h_pred.exp() * down_factor)
        d = self.decode_d(d_pred, cam2img)
        xyz = self.decode_xyz(uv, d, img2cam)
        ry = self.decode_ry(alpha_4bin_pred, alpha_pred, xyz, K_out)
        bbox3d = torch.cat((xyz, lhw_pred, ry), dim=1)
        if tranpose:
            bbox3d = bbox3d.transpose(1, 2)
        return bbox3d

    def decode_lhw(self, lhw_pred, labels):
        '''
        Args:
            lhw_pred:[b,3,k]
            labels:[b,k]
        Returns:
            lhw_pred:[b,3,k]
        '''
        batch_size, _, topk = lhw_pred.size()
        lhw_pred = lhw_pred.exp() * torch.gather(self.base_dims.expand(batch_size, 3, 3), 2,
                                                 labels.expand(batch_size, 3, topk))
        return lhw_pred

    def decode_d(self, d_pred, cam2img):
        '''

        Args:
            d_pred: [b,1,k]
            cam2img: [b,4,4]

        Returns:
            d: [b,1,k]
        '''
        batch_size, _, topk = d_pred.size()
        d = cam2img[:, 1, 1].view(batch_size, 1, 1) * (d_pred * self.base_depth[1] + self.base_depth[0])
        return d

    def decode_ry(self, alpha_4bin_pred, alpha_pred, xyz, K_out):
        '''

        Args:
            alpha_4bin_pred: [b,4,k] or [b,1,k]
            alpha_pred: [b,1,k] or [b,2,k]
            xyz: [b,3,k]

        Returns:
            ry:[b,1,k]
        '''
        if alpha_4bin_pred.shape[1] == 4:
            _, bin = torch.max(alpha_4bin_pred, dim=1, keepdim=False)
        else:
            bin = alpha_4bin_pred.squeeze(1)
        bin0 = (bin == 0).float()
        bin1 = (bin == 1).float()
        bin2 = (bin == 2).float()
        bin3 = (bin == 3).float()
        if alpha_pred.shape[1] == 2:
            alpha_pred = torch.atan2(alpha_pred[:, 0, :], alpha_pred[:, 1, :])
        else:
            alpha_pred = alpha_pred.squeeze(1)
        alpha = alpha_pred * (bin0 + bin2 - bin1 - bin3) + np.pi * (bin0 + bin3)
        ry = (torch.atan2(xyz[:, 2, :] + K_out[:, 2].unsqueeze(1),
                          xyz[:, 0, :] + K_out[:, 0].unsqueeze(1)) + alpha).unsqueeze(1)
        # ry = alpha.unsqueeze(1)
        ry = (ry + np.pi) % (2 * np.pi) - np.pi
        return ry

    def decode_xyz(self, uv, d, img2cam, d_heatmap=None):
        '''

        Args:
            uv: [b,2,k]
            d:  [b,1,k]
            img2cam:[b,4,4]

        Returns:
            xyz: [b, 3, k]
        '''
        if d_heatmap is not None:
            centers2d_img = torch.cat((uv * d_heatmap, d_heatmap, self.constant_1.expand_as(d)), dim=1)
            xy = torch.bmm(img2cam[:, :2, :], centers2d_img)
            z = d + img2cam[:, 2::8, 3::8]
            xyz = torch.cat([xy, z], dim=1)
        else:
            centers2d_img = torch.cat((uv * d, d, self.constant_1.expand_as(d)), dim=1)
            xyz = torch.bmm(img2cam, centers2d_img)
            xyz = xyz[:, :3, :]
        return xyz

    def box_bev_nms(self, bboxes3d, scores, labels):
        x1, y1, z1, l1, h1, w1, ry1 = torch.split(bboxes3d, 1, dim=1)
        bboxes_for_nms = torch.cat((x1, z1, l1, w1, ry1), dim=1)
        bboxes_for_nms = xywhr2xyxyr(bboxes_for_nms)
        keep = nms_gpu(bboxes_for_nms, scores, self.test_cfg.nms.iou_threshold)
        labels = labels[keep]
        scores = scores[keep]
        bboxes3d = bboxes3d[keep, :]
        return bboxes3d, scores, labels, keep

    def box2d_nms(self, bboxes, scores, labels):
        out_bboxes, keep = batched_nms(bboxes, scores, labels, self.test_cfg.nms)
        out_labels = labels[keep]
        return out_bboxes, out_labels, keep

    def bbox2d_to_3d(self, xyxy, h1h2, lhw, face_label, K_in_inv, fv, w):
        N = xyxy.size()[0]
        xyxy = xyxy.view(N, 2, 2)
        H, W, L = torch.split(lhw, [1, 1, 1], dim=1)
        K_in_inv, bias = torch.split(K_in_inv.clone(), (3, 1), dim=1)
        # K_in_inv = K_in.inverse()
        K_in_inv = K_in_inv.T.unsqueeze(0)
        d = fv * H / h1h2
        d = d.unsqueeze(2)
        udvd = xyxy * d
        uvd = torch.cat((udvd, d), dim=2)
        temp = uvd - bias.view(1, 1, 3)
        xyz = torch.matmul(temp, K_in_inv)
        xyz1, xyz2 = torch.split(xyz, 1, dim=1)
        delta_xyz = xyz2 - xyz1
        delta_xyz = delta_xyz.squeeze(1)
        delta_xz = delta_xyz[:, ::2]
        # delta_x, delta_y, delta_z = torch.split(delta_xyz, 1, dim=1)
        delta_xz_norm = torch.norm(delta_xz, dim=1, keepdim=True, p=2)
        theta = torch.sign(delta_xz[:, 1]) * torch.acos(delta_xz[:, 0] / delta_xz_norm.squeeze(1))
        theta = -theta - (face_label - 1) * math.pi / 2
        theta = torch.where(theta > math.pi, theta - 2 * math.pi, theta)
        theta = torch.where(theta < -math.pi, theta + 2 * math.pi, theta)
        r_y = theta.unsqueeze(1)
        truncation_left = (xyxy[:, 0, 0] <= 2).unsqueeze(1)
        truncation_right = (xyxy[:, 1, 0] >= w - 3).unsqueeze(1)
        face_label = face_label.unsqueeze(1)
        face_13 = (face_label == 1) + (face_label == 3)
        face_24 = ~face_13
        W_norm = 0.5 * W / delta_xz_norm
        L_norm = 0.5 * L / delta_xz_norm
        temp1 = face_13 * W_norm + face_24 * L_norm
        temp2 = face_13 * L_norm + face_24 * W_norm
        temp3 = truncation_right * (temp2 - 0.5) + truncation_left * (0.5 - temp2)
        delta_zx = delta_xz[:, self.constant_1_0] * self.constant_neg1_1
        mean_xyz = torch.mean(xyz, dim=1, keepdim=False)
        mean_xyz[:, ::2] += delta_xz * temp3 + delta_zx * temp1
        temp4 = (truncation_left == 0) * (truncation_right == 0)
        update_w = temp4 * face_24
        update_l = temp4 * face_13
        W = W * (~update_w) + delta_xz_norm * update_w
        L = L * (~update_l) + delta_xz_norm * update_l
        return mean_xyz, H, W, L, r_y

    def bbox3d_to_2d(self, xyz, lhw, r_y, K_in):
        if xyz.size()[0] > 0:
            N = r_y.size()[0]
            K_in, bias = torch.split(K_in, (3, 1), dim=1)
            cos_ry = (torch.cos(r_y))
            sin_ry = (torch.sin(r_y))
            zero = self.constant_0.expand(N, 1)
            one = self.constant_1.expand(N, 1)
            rot = torch.cat((cos_ry, zero, sin_ry, zero, one, zero, -sin_ry, zero, cos_ry), dim=1).view([N, 3, 3])
            lhw = lhw[:, self.constant_2_0_1]
            temp2 = lhw.unsqueeze(2) * self.cor
            temp4 = torch.matmul(rot, temp2) + xyz.unsqueeze(2)
            torch.clamp_min_(temp4[:, 2], 1e-4)
            uvd = torch.matmul(K_in.unsqueeze(0), temp4) + bias.unsqueeze(0)
            uv, d = torch.split(uvd, (2, 1), dim=1)
            uv = uv / d
            xy_min, _ = torch.min(uv, dim=2, keepdim=False)
            xy_max, _ = torch.max(uv, dim=2, keepdim=False)
            return torch.cat((xy_min, xy_max), dim=1)
        else:
            return xyz.new_zeros((0, 4))
