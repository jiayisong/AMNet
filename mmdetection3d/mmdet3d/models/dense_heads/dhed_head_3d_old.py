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
from mmdet.models.utils import gaussian_radius, gen_gaussian_target

from mmcv.ops import box_iou_rotated
from mmcv.runner import BaseModule
from abc import ABCMeta, abstractmethod

@HEADS.register_module()
class DHEDHead3d(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channel,
                 strides,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap,
                 loss_hwl,
                 loss_size,
                 loss_xyzxyz,
                 loss_iou,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(DHEDHead3d, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.strides = strides
        self.heatmap_head = nn.ModuleList()
        self.heatmap_stem = nn.ModuleList()
        self.hwl_stem = nn.ModuleList()
        self.hwl_head = nn.ModuleList()
        self.size_stem = nn.ModuleList()
        self.iou_head = nn.ModuleList()
        self.size_head = nn.ModuleList()
        self.x = []
        self.y = []
        for i, inp in enumerate(in_channel):
            self.heatmap_stem.append(nn.Sequential(self.conv3x3(inp, feat_channel),
                                                   self.conv3x3(feat_channel, feat_channel)))
            self.heatmap_head.append(self.conv1x1(feat_channel, self.num_classes))
            self.hwl_stem.append(
                nn.Sequential(self.conv3x3(inp, feat_channel), self.conv3x3(feat_channel, feat_channel)))
            self.hwl_head.append(self.conv1x1(feat_channel, 3))
            self.size_stem.append(nn.Sequential(self.conv3x3(inp, feat_channel),
                                                self.conv3x3(feat_channel, feat_channel)))
            self.iou_head.append(self.conv1x1(feat_channel, 1))
            self.size_head.append(self.conv1x1(feat_channel, 6))
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
        self.loss_hwl = build_loss(loss_hwl)
        self.loss_size = build_loss(loss_size)
        self.loss_iou = build_loss(loss_iou)
        self.loss_xyzxyz = build_loss(loss_xyzxyz)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.register_buffer('cor',
                             torch.tensor([[[0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], [0, 0, 0, 0, -1, -1, -1, -1],
                                            [0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5]]], dtype=torch.float32))
        self.register_buffer('constant_neg1_1', torch.tensor([[-1, 1]], dtype=torch.float32))
        # self.register_buffer('constant_1_neg1', torch.tensor([[[[1]], [[-1]]]], dtype=torch.float32))
        self.register_buffer('constant_1_0', torch.tensor([1, 0], dtype=torch.long))
        self.register_buffer('constant_2_0_1', torch.tensor([2, 0, 1], dtype=torch.long))
        self.register_buffer('constant_0', torch.tensor([[0]], dtype=torch.float32))
        self.register_buffer('constant_1', torch.tensor([[1]], dtype=torch.float32))

    def conv3x3(self, in_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return layer

    def conv1x1(self, in_channel, out_channel):
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        for heatmap_head, size_head, iou_head, hwl_head in zip(self.heatmap_head, self.size_head, self.iou_head,
                                                               self.hwl_head):
            heatmap_head.bias.data.fill_(bias_init_with_prob(0.01))
            size_head.bias.data[:3].fill_(2)
            size_head.bias.data[3:].fill_(0.5)
            iou_head.bias.data.fill_(bias_init_with_prob(0.01))
            hwl_head.bias.data.fill_(0.5)
        # self.heatmap_head[-1].weight.data.fill_(0.01)
        # self.size_head[-1].weight.data.fill_(0.01)
        # self.hwl_head[-1].weight.data.fill_(0.01)

    def forward_train(self, x, img_metas, **kwargs):
        outs = self(x)
        losses = self.loss(*outs, **kwargs)
        return losses

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.heatmap_stem, self.heatmap_head, self.hwl_stem,
                           self.hwl_head,
                           self.size_stem, self.iou_head, self.size_head)

    def simple_test(self, feats, img_metas, **kwargs):
        return self.simple_test_bboxes(feats, img_metas, **kwargs)

    def simple_test_bboxes(self, feats, img_metas, **kwargs):
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, **kwargs)
        return results_list

    def forward_single(self, feat, heatmap_stem, heatmap_head, hwl_stem, hwl_head, size_stem, iou_head, size_head):
        heatmap = heatmap_stem(feat)
        center_heatmap_pred = heatmap_head(heatmap).sigmoid()
        heatmap = hwl_stem(feat)
        hwl_pred = hwl_head(heatmap)
        size_pred = size_stem(feat)
        iou_pred = iou_head(size_pred).sigmoid()
        size_pred = size_head(size_pred)
        return center_heatmap_pred, hwl_pred, size_pred, iou_pred

    def uvh2xyz(self, hwl_preds, size_preds, K_in_inv, fv, x, y, stride):
        batch_size, _, height, width = hwl_preds.size()
        w, h1h2, offset_x, offset_y1y2 = torch.split(size_preds, [1, 2, 1, 2], dim=1)
        H, W, L = torch.split(hwl_preds, 1, dim=1)
        w = w.exp()
        h1h2 = h1h2.exp()
        xc = x + offset_x
        d1d2 = H.exp() / h1h2 * fv.view(batch_size, 1, 1, 1) / stride
        x1x2 = (xc + self.constant_neg1_1.view(1, 2, 1, 1) * w * 0.5) * stride * d1d2
        y1y2 = (y + offset_y1y2 + h1h2 * 0.5) * stride * d1d2
        uvduvd = torch.stack([x1x2, y1y2, d1d2], dim=4).unsqueeze(5)
        K_in_inv, bias = torch.split(K_in_inv.clone(), (3, 1), dim=2)
        K_in_inv = K_in_inv.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        xyzxyz = torch.matmul(K_in_inv, uvduvd - bias.view(batch_size, 1, 1, 1, 3, 1))
        xyzxyz = xyzxyz.permute(0, 1, 4, 2, 3, 5).reshape(batch_size, 6, height, width)
        return xyzxyz

    @force_fp32(apply_to=('center_heatmap_preds', 'hwl_preds', 'size_preds', 'iou_pred'))
    def loss(self, center_heatmap_preds, hwl_preds, size_preds, iou_pred,
             center_heatmap_pos, center_heatmap_neg, size_heatmap, hwl_heatmap, xyzxyz_heatmap, size_mask, P2_inv, fv
             ):
        assert len(center_heatmap_preds) == len(hwl_preds) == len(size_preds) == len(center_heatmap_pos) == len(
            center_heatmap_neg) == len(size_heatmap) == len(size_mask)
        xyzxyzs = []
        for hwl_pred, size_pred, x, y, s in zip(hwl_preds, size_preds, self.x, self.y, self.strides):
            # for hwl_pred, size_pred, x, y, s in zip(hwl_heatmap, size_heatmap, self.x, self.y, self.strides):
            xyzxyzs.append(self.uvh2xyz(hwl_pred, size_pred, P2_inv, fv, getattr(self, x), getattr(self, y), s))
        loss_center_heatmap, loss_center_heatmap_show = self.loss_center_heatmap(center_heatmap_preds,
                                                                                 center_heatmap_pos, center_heatmap_neg)
        loss_size, loss_size_show = self.loss_size(size_preds, size_heatmap, size_mask)
        loss_hwl, loss_hwl_show = self.loss_hwl(hwl_preds, hwl_heatmap, size_mask)
        loss_xyzxyz, loss_xyzxyz_show = self.loss_xyzxyz(xyzxyzs, xyzxyz_heatmap, size_mask)
        # Wloss_iou, loss_iou_show = self.loss_iou(iou_pred, size_preds, size_heatmap, size_mask)

        loss = loss_center_heatmap
        loss = loss + loss_hwl
        # loss = loss + loss_xyzxyz
        # loss = loss + loss_iou
        loss = loss + loss_size
        outputs = dict(loss=loss)
        for key, value in loss_center_heatmap_show.items():
            outputs[f'heatmap_{key}'] = value
        # for key, value in loss_iou_show.items():
        #    outputs[f'iou_{key}'] = value
        for key, value in loss_hwl_show.items():
            outputs[f'hwl_{key}'] = value
        for key, value in loss_xyzxyz_show.items():
            outputs[f'xyzxyz_{key}'] = value
        for key, value in loss_size_show.items():
            outputs[f'size_{key}'] = value
        return outputs

    # @profile
    def get_bboxes(self, center_heatmap_preds, hwl_preds, size_preds, iou_preds, img_metas, P2,  P2_inv,
                   fv,#scale_factor,
                   # center_heatmap_pos, size_heatmap, hwl_heatmap,xyzxyz_heatmap,#xyzxyz,bbox8d,xyz,
                   rescale=False):
        #a = center_heatmap_pos[0][0], size_heatmap[0][0], hwl_heatmap[0][0], xyzxyz_heatmap[0][0],# xyzxyz, bbox8d, xyz
        #b = [i.cpu().numpy() for i in a]
        assert len(center_heatmap_preds) == len(hwl_preds) == len(size_preds) == len(iou_preds)
        # center_heatmap_preds, size_preds, hwl_preds = center_heatmap_pos, size_heatmap, hwl_heatmap
        batch_det_xyxys, batch_det_h1h2s, batch_scores, batch_labels, batch_hwls = [], [], [], [], []
        for center_heatmap_pred, hwl_pred, size_pred, iou_pred in zip(center_heatmap_preds, hwl_preds,
                                                                      size_preds, iou_preds):
            # center_heatmap_pred = center_heatmap_pred * iou_pred
            batch_det_xyxy, batch_det_h1h2, batch_score, batch_label, batch_hwl = self.decode_heatmap(
                center_heatmap_pred, hwl_pred,
                size_pred,
                img_metas[0]['batch_input_shape'])
            if False:
                batch_det_xyxy = batch_det_xyxy / scale_factor[0].unsqueeze(1)
                batch_det_h1h2 = batch_det_h1h2 / scale_factor[0][:, 1].unsqueeze(1).unsqueeze(1)
            batch_det_xyxys.append(batch_det_xyxy)
            batch_scores.append(batch_score)
            batch_labels.append(batch_label)
            batch_det_h1h2s.append(batch_det_h1h2)
            batch_hwls.append(batch_hwl)
        batch_det_xyxys = torch.cat(batch_det_xyxys, dim=1)
        batch_det_h1h2s = torch.cat(batch_det_h1h2s, dim=1)
        batch_hwls = torch.cat(batch_hwls, dim=1)
        batch_scores = torch.cat(batch_scores, dim=1)
        batch_labels = torch.cat(batch_labels, dim=1)
        det_results = []
        for (det_xyxy, det_h1h2, det_scores, det_labels, det_hwls, K_in, K_in_inv, Fv, img_meta) in zip(batch_det_xyxys,
                                                                                                        batch_det_h1h2s,
                                                                                                        batch_scores,
                                                                                                        batch_labels,
                                                                                                        batch_hwls,
                                                                                                        P2[0],
                                                                                                        P2_inv[0],
                                                                                                        fv[0],
                                                                                                        img_metas):
            det_result = {}
            h, w = img_meta['ori_shape']
            N = det_labels.size()[0]
            det_face_label = det_labels % 4
            det_class_label = det_labels // 4
            xyz, H, W, L, r_y = self.bbox2d_to_3d(det_xyxy, det_h1h2, det_hwls, det_face_label, K_in_inv, Fv, w)
            x, y, z = torch.split(xyz, 1, dim=1)
            hwl = torch.cat((H, W, L), dim=1)
            iou, _ = self.box3d_iou(x, y, z, H, W, L, r_y)
            #iou_debug = iou.cpu().numpy()
            iou = iou.unsqueeze(0).unsqueeze(0)
            mask = iou > self.test_cfg.iou_threshold
            temp = torch.nn.functional.pad(mask, (0, 0, N, 0))
            temp = temp[:, :, :-1, :]
            temp = torch.nn.functional.max_pool2d(temp.float(), (N, 1), stride=1, padding=0)
            mask = mask * (1 - temp) * iou
            weight = (det_scores.unsqueeze(0) * mask.squeeze(0).squeeze(0)).unsqueeze(2)

            weight_sum = torch.sum(weight, dim=1, keepdim=True)
            available_bbox = (weight_sum > 0).squeeze(1).squeeze(1)
            weight = weight / weight_sum
            #weight_debug = weight.squeeze(2).cpu().numpy()
            xyz = torch.sum(xyz.unsqueeze(0) * weight, dim=1, keepdim=False)
            hwl = torch.sum(hwl.unsqueeze(0) * weight, dim=1, keepdim=False)
            r_y = torch.sum(r_y.unsqueeze(0) * weight, dim=1, keepdim=False)

            det_result['label'] = det_class_label[available_bbox]
            det_result['score'] = det_scores[available_bbox]
            det_result['hwl'] = hwl[available_bbox, :]
            det_result['r_y'] = r_y[available_bbox, :]
            det_result['xyz'] = xyz[available_bbox, :]
            det_result['bbox'] = self.bbox3d_to_2d(det_result['xyz'], det_result['hwl'], det_result['r_y'], K_in)
            det_result['label'] = det_result['label'].cpu().numpy()
            det_result['score'] = det_result['score'].cpu().numpy()
            det_result['hwl'] = det_result['hwl'].cpu().numpy()
            det_result['r_y'] = det_result['r_y'].cpu().numpy()
            det_result['xyz'] = det_result['xyz'].cpu().numpy()
            det_result['bbox'] = det_result['bbox'].cpu().numpy()
            det_result['bbox'][:, ::2] = np.clip(det_result['bbox'][:, ::2], 0, w - 1)
            det_result['bbox'][:, 1::2] = np.clip(det_result['bbox'][:, 1::2], 0, h - 1)
            det_results.append(det_result)
        return det_results

    def bbox2d_to_3d(self, xyxy, h1h2, hwl, face_label, K_in_inv, fv, w):
        N = xyxy.size()[0]
        xyxy = xyxy.view(N, 2, 2)
        H, W, L = torch.split(hwl, [1, 1, 1], dim=1)
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

    def bbox3d_to_2d(self, xyz, hwl, r_y, K_in):
        if xyz.size()[0] > 0:
            N = r_y.size()[0]
            K_in, bias = torch.split(K_in, (3, 1), dim=1)
            cos_ry = (torch.cos(r_y))
            sin_ry = (torch.sin(r_y))
            zero = self.constant_0.expand(N, 1)
            one = self.constant_1.expand(N, 1)
            rot = torch.cat((cos_ry, zero, sin_ry, zero, one, zero, -sin_ry, zero, cos_ry), dim=1).view([N, 3, 3])
            lhw = hwl[:, self.constant_2_0_1]
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

    @staticmethod
    def decode_heatmap(center_heatmap_pred, hwl_pred, size_pred, img_shape, k=100):
        batch_size, class_num, height, width = center_heatmap_pred.shape
        inp_h, inp_w = img_shape
        down_factor = inp_h / height
        hw = height * width
        topk_scores, topk_inds = torch.topk(center_heatmap_pred.view(batch_size, -1), k)
        topk_clses = topk_inds // hw
        topk_inds = topk_inds % hw
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).float()
        batch_topk_labels = topk_clses
        batch_scores = topk_scores
        batch_index = topk_inds.unsqueeze(1)
        hwl_pred = hwl_pred.view(batch_size, hwl_pred.size(1), hw)
        batch_hwls = hwl_pred.gather(dim=2, index=batch_index.expand((batch_size, hwl_pred.size(1), k)))
        batch_hwls = batch_hwls.transpose(1, 2).exp()
        size_pred = size_pred.view(batch_size, size_pred.size(1), hw)
        size = size_pred.gather(dim=2, index=batch_index.expand((batch_size, size_pred.size(1), k)))
        size = size.transpose(1, 2)
        x = topk_xs
        y = topk_ys
        h1 = size[:, :, 1].exp()
        h2 = size[:, :, 2].exp()
        x1 = x + size[:, :, 3] - 0.5 * size[:, :, 0].exp()
        y2 = y + size[:, :, 4] + 0.5 * h1
        x3 = x + size[:, :, 3] + 0.5 * size[:, :, 0].exp()
        y4 = y + size[:, :, 5] + 0.5 * h2
        h1h2 = torch.stack([h1, h2], dim=2) * down_factor
        xyxy = torch.stack([x1, y2, x3, y4], dim=2) * down_factor
        return xyxy, h1h2, batch_scores, batch_topk_labels, batch_hwls

    def box3d_iou(self, x, y, z, H, W, L, r_y):
        bboxes = torch.cat((x, z, L, W, r_y), dim=1)
        pred_area = L * W
        inters = box_iou_rotated(bboxes, bboxes, mode='iof', aligned=False) * pred_area
        y_min = y - H
        y_max = y
        inter_h = torch.minimum(y_max, y_max.T) - torch.maximum(y_min, y_min.T)
        inters_3d = inters * inter_h * (inter_h > 0)
        pred_area_3d = pred_area * H
        return inters / (pred_area + pred_area.T - inters), inters_3d / (pred_area_3d + pred_area_3d.T - inters_3d)
