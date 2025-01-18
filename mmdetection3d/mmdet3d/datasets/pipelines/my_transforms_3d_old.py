import math
from mmdet.core import multi_apply
from mmdet.datasets.builder import PIPELINES
import numpy as np
import mmcv
from numpy import random
import torch
import cv2
from ...core.bbox.structures.utils import points_cam2img


def clip_bbox8d(bbox, x_min, x_max, y_min, y_max):
    # x1, y1, x2, y2, x3, y3, x4, y4 = np.split(bbox, [1,2,3,4,5,6,7], axis=1)
    bbox = bbox.copy()
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    x3 = bbox[:, 4]
    y3 = bbox[:, 5]
    x4 = bbox[:, 6]
    y4 = bbox[:, 7]
    keep1 = x1 < x_max
    keep2 = x3 > x_min
    keep3 = (x1 < x_min) * keep2
    keep4 = (x3 > x_max) * keep1
    temp1 = (x_min - x1[keep3]) / (x3[keep3] - x1[keep3])
    temp2 = (x_min - x3[keep3]) / (x1[keep3] - x3[keep3])
    y1[keep3] = y3[keep3] * temp1 + y1[keep3] * temp2
    y2[keep3] = y4[keep3] * temp1 + y2[keep3] * temp2
    x1[keep3] = x_min
    x2[keep3] = x_min
    temp3 = (x_max - x1[keep4]) / (x3[keep4] - x1[keep4])
    temp4 = (x_max - x3[keep4]) / (x1[keep4] - x3[keep4])
    y3[keep4] = y3[keep4] * temp3 + y1[keep4] * temp4
    y4[keep4] = y4[keep4] * temp3 + y2[keep4] * temp4
    x3[keep4] = x_max
    x4[keep4] = x_max
    keep5 = np.minimum(y1, y3) < y_max
    keep6 = np.maximum(y2, y4) > y_min
    keep = keep1 * keep2 * keep5 * keep6
    return bbox[keep, :], keep


@PIPELINES.register_module()
class Bbox3dTo2d:
    def __call__(self, results):
        N = results['r_y'].shape[0]
        img_h, img_w = results['img_shape']
        temp1 = np.array([[0.5, 0.5, -0.5, -0.5], [0, 0, 0, 0], [0.5, -0.5, -0.5, 0.5]], dtype=np.float32)
        cos_ry = (np.cos(results['r_y']))
        sin_ry = (np.sin(results['r_y']))
        zero = np.zeros_like(cos_ry)
        one = np.ones_like(cos_ry)
        z = results['xyz'][:, 2]
        rot = np.concatenate((cos_ry, zero, sin_ry, results['xyz'][:, [0]],
                              zero, one, zero, results['xyz'][:, [1]],
                              -sin_ry, zero, cos_ry, z[:, None],
                              zero, zero, zero, one), axis=1).reshape([N, 4, 4])
        temp2 = results['lhw'][:, [2, 0, 1], None] * temp1
        temp3 = np.ones_like(rot)
        temp3[:, :3, :] = temp2
        temp4 = np.matmul(rot, temp3)
        temp4[:, 2, :] = np.clip(temp4[:, 2, :], 1e-4, np.inf)
        uvd = np.matmul(results['P2'][None, :, :], temp4)
        d = uvd[:, 2, :]
        uv = uvd[:, :2, :] / d[:, None, :]
        u = uv[:, 0, :]
        h_2d = results['P2'][1, 1] * results['lhw'][:, 0, None] / d
        uv2 = np.concatenate((uv[:, [0], :], uv[:, [1], :] - h_2d[:, None, :], uv), axis=1)
        u_minarg = np.argmin(u, axis=1)
        u_maxarg = np.argmax(u, axis=1)
        show2side = np.abs(u_maxarg - u_minarg) == 2
        show1side = ~show2side
        uv2_min = uv2[np.arange(N), :, u_minarg]
        uv2_max = uv2[np.arange(N), :, u_maxarg]
        bbox8d_1side = np.concatenate((uv2_min[show1side, :], uv2_max[show1side, :]), axis=1)
        face_label_1side = u_maxarg[show1side]
        class_label_1side = results['gt_labels'][show1side]
        lhw_1side = results['lhw'][show1side]
        u_midarg = u_minarg - 1
        u_midarg[u_midarg == -1] = 3
        uv2_mid = uv2[np.arange(N), :, u_midarg]
        bbox8d_2side = [np.concatenate((uv2_min[show2side, :], uv2_mid[show2side, :]), axis=1),
                        np.concatenate((uv2_mid[show2side, :], uv2_max[show2side, :]), axis=1)]
        z = np.concatenate([z[show1side], z[show2side], z[show2side]], axis=0)
        order = np.argsort(z)[::-1]
        bbox8d_2side = np.concatenate(bbox8d_2side, axis=0)
        face_label_2side = np.concatenate((u_midarg[show2side], u_maxarg[show2side]), axis=0)
        class_label_2side = np.concatenate((results['gt_labels'][show2side], results['gt_labels'][show2side]), axis=0)
        lhw_2side = np.concatenate((results['lhw'][show2side], results['lhw'][show2side]), axis=0)
        bbox8d = np.concatenate((bbox8d_1side, bbox8d_2side), axis=0)
        facelabel = np.concatenate((face_label_1side, face_label_2side), axis=0)
        classlabel = np.concatenate((class_label_1side, class_label_2side), axis=0)
        bbox8d_lhw = np.concatenate((lhw_1side, lhw_2side), axis=0)
        bbox8d = bbox8d[order, :]
        facelabel = facelabel[order]
        classlabel = classlabel[order]
        bbox8d_lhw = bbox8d_lhw[order, :]
        bbox8d, keep = clip_bbox8d(bbox8d, 0, img_w - 1, 0, img_h - 1)
        results['bbox8d'] = bbox8d
        results['facelabel'] = facelabel[keep]
        results['classlabel'] = classlabel[keep]
        results['bbox8d_lhw'] = bbox8d_lhw[keep, :]
        results['bbox_fields'].append('bbox8d')
        return results


@PIPELINES.register_module()
class MakeHeatMap3dTwoStage():
    def __init__(self, size, label_num, max_num_pre_img, down_factor, kernel_size, size_distribution, base_depth,
                 base_alpha, alpha_type,
                 base_dims, center_type, beta, free_loss, iou_heat, train_without_far,
                 train_without_ignore, train_without_outbound, train_without_small):
        self.size = size
        self.label_num = label_num
        self.max_num_pre_img = max_num_pre_img
        self.down_factor = down_factor
        self.center_type = center_type
        self.train_without_ignore = train_without_ignore
        self.train_without_outbound = train_without_outbound
        self.train_without_small = train_without_small
        self.train_without_far = train_without_far
        self.k_heat = kernel_size
        self.beta = beta
        self.free_loss = free_loss
        self.iou_heat = iou_heat
        self.size_distribution = size_distribution
        self.base_depth = base_depth
        self.base_alpha = base_alpha
        self.alpha_type = alpha_type
        self.base_dims = torch.tensor(base_dims)

    def h_sin(self, x):
        return x + (np.pi - 2 * x) * (x > (np.pi / 2)) + (-np.pi - 2 * x) * (x < (-np.pi / 2))

    def h_cos(self, x):
        return np.pi / 2 - torch.abs(x)

    def gen_init_tensor(self, down_factor):
        heat_map_size = torch.div(torch.tensor(self.size), down_factor, rounding_mode='trunc')
        center_heatmap_pos = torch.zeros([self.label_num, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        center_heatmap_neg = torch.ones([self.label_num, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        center_heatmap_temp = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        index_heatmap = (self.max_num_pre_img - 1) * torch.ones([1, heat_map_size[0], heat_map_size[1]],
                                                                dtype=torch.long)
        size_mask = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        size_heatmap = torch.ones([4, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)

        x = torch.arange(0, heat_map_size[1])
        x = x.unsqueeze(0).expand([heat_map_size[0], heat_map_size[1]])
        y = torch.arange(0, heat_map_size[0])
        y = y.unsqueeze(1).expand([heat_map_size[0], heat_map_size[1]])
        return heat_map_size, center_heatmap_pos, center_heatmap_neg, center_heatmap_temp, size_mask, size_heatmap, index_heatmap, x, y

    def __call__(self, results):
        size_distribution = np.array(self.size_distribution)
        img_shape = np.array(results['img_shape'][:2], dtype=np.float32)
        down_factor = np.array(self.down_factor)
        k_heat = self.k_heat
        cam2img = torch.tensor(results['cam2img'])
        heat_map_sizes, center_heatmap_poses, center_heatmap_negs, center_heatmap_temps, size_masks, size_heatmaps, index_heatmaps, xs, ys = multi_apply(
            self.gen_init_tensor, self.down_factor)

        _, indices = torch.sort(results['gt_bboxes_3d'].gravity_center[:, 2], descending=True)

        results['gt_bboxes_3d'] = results['gt_bboxes_3d'][indices, :]
        depths = points_cam2img(results['gt_bboxes_3d'].gravity_center, cam2img, True)[:, 2].numpy()
        indices = indices.numpy()
        if len(indices.shape) == 0:
            indices = indices[None, :]
        results['gt_labels'] = results['gt_labels'][indices]
        results['gt_bboxes'] = results['gt_bboxes'][indices, :]
        indices = (((results['gt_bboxes'][:, 2] - results['gt_bboxes'][:, 0]) > self.train_without_small[1]) * (
                (results['gt_bboxes'][:, 3] - results['gt_bboxes'][:, 1]) > self.train_without_small[0])) * (
                          depths <= self.train_without_far)
        if self.train_without_outbound and self.center_type == '3d':
            gt_bboxes_3d = results['gt_bboxes_3d']
            center_3ds = points_cam2img(gt_bboxes_3d.gravity_center, cam2img, True)
            temp = (center_3ds[:, 0] >= 0) * (center_3ds[:, 1] >= 0) * (center_3ds[:, 0] <= self.size[1] - 1) * (
                    center_3ds[:, 1] <= self.size[0] - 1)
            indices = indices * temp.numpy()
        results['gt_bboxes_ignore'] = np.concatenate(
            (results['gt_bboxes_ignore'], results['gt_bboxes'][~indices, :]), axis=0)
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'][indices, :]
        results['gt_labels'] = results['gt_labels'][indices]
        results['gt_bboxes'] = results['gt_bboxes'][indices, :]
        bbox_num = len(results['gt_labels'])

        K_out = results['K_out']
        boxes_ignore = results['gt_bboxes_ignore']
        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels = results['gt_labels']
        gt_bboxes = results['gt_bboxes']
        lhws = gt_bboxes_3d.dims
        xyzs = gt_bboxes_3d.gravity_center
        corners = gt_bboxes_3d.corners
        center_3ds = points_cam2img(gt_bboxes_3d.gravity_center, cam2img, True)
        corners_2d = points_cam2img(corners.view(-1, 3), cam2img, True)
        depths = center_3ds[:, 2]
        center_3ds = center_3ds[:, :2]
        corners_depths = corners_2d[:, 2].view(-1, 8)
        corners_2d = corners_2d[:, :2].reshape(-1, 16)
        # h2ds = 2 * (points_cam2img(gt_bboxes_3d.bottom_center, cam2img)[:, 1] - center_3ds[:, 1])
        rys = gt_bboxes_3d.tensor[..., 6]
        alphas = rys - torch.atan2(gt_bboxes_3d.tensor[..., 2] + K_out[2], gt_bboxes_3d.tensor[..., 0] + K_out[0])
        alphas = (alphas + np.pi) % (2 * np.pi) - np.pi
        alpha4bin_heatmap = torch.zeros([self.max_num_pre_img, 4], dtype=torch.long)
        if self.alpha_type == 'my4bin':
            merge = 0
            alpha4bin = torch.stack([
                (alphas >= -merge) * (alphas < merge + np.pi / 2),
                (alphas >= np.pi / 2 - merge) + (alphas < -np.pi + merge),
                (alphas >= np.pi - merge) + (alphas < -np.pi / 2 + merge),
                (alphas >= -np.pi / 2 - merge) * (alphas < merge)
            ], 1)
            alphas = torch.sum(torch.stack([
                # (alphas - np.pi / 4), (alphas - 3 * np.pi / 4), (alphas + 3 * np.pi / 4), (alphas + np.pi / 4),
                (alphas - np.pi / 4), -(alphas - 3 * np.pi / 4), (alphas + 3 * np.pi / 4), -(alphas + np.pi / 4),
            ], 1) * alpha4bin, dim=1, keepdim=True)
            alpha_heatmap = torch.zeros([self.max_num_pre_img, 1], dtype=torch.float32)
            alpha_heatmap[:bbox_num, :] = (alphas - self.base_alpha[0]) / self.base_alpha[1]
            alpha4bin_heatmap[:bbox_num, :] = alpha4bin
        elif self.alpha_type == '4bin':
            merge = 0
            alpha4bin = torch.stack([
                (alphas >= -merge) * (alphas < merge + np.pi / 2),
                (alphas >= np.pi / 2 - merge) + (alphas < -np.pi + merge),
                (alphas >= np.pi - merge) + (alphas < -np.pi / 2 + merge),
                (alphas >= -np.pi / 2 - merge) * (alphas < merge)
            ], 1)
            alphas = torch.stack([
                (alphas - np.pi / 4), (alphas - 3 * np.pi / 4), (alphas + 3 * np.pi / 4), (alphas + np.pi / 4),
                # (alphas - np.pi / 4), -(alphas - 3 * np.pi / 4), (alphas + 3 * np.pi / 4), -(alphas + np.pi / 4),
            ], 1)
            alphas = (alphas + np.pi) % (2 * np.pi) - np.pi
            alpha_heatmap = torch.zeros([self.max_num_pre_img, 4], dtype=torch.float32)
            alpha_heatmap[:bbox_num, :] = (alphas - self.base_alpha[0]) / self.base_alpha[1]
            alpha4bin_heatmap[:bbox_num, :] = alpha4bin

        elif self.alpha_type == 'sincos':
            alphas = torch.stack((torch.sin(alphas), torch.cos(alphas)), dim=1)
            alpha_heatmap = torch.zeros([self.max_num_pre_img, 2], dtype=torch.float32)
            alpha_heatmap[:bbox_num, :] = (alphas - self.base_alpha[0]) / self.base_alpha[1]
        elif self.alpha_type == 'hsinhcos':
            alphas = torch.stack((self.h_sin(alphas), self.h_cos(alphas)), dim=1)
            alpha_heatmap = torch.zeros([self.max_num_pre_img, 2], dtype=torch.float32)
            alpha_heatmap[:bbox_num, :] = (alphas - self.base_alpha[0]) / self.base_alpha[1]

        gt_labels = torch.from_numpy(gt_labels)
        cls_heatmap_pos = torch.zeros([self.max_num_pre_img, self.label_num], dtype=torch.float32)
        cls_heatmap_pos[:bbox_num, :] = torch.scatter(cls_heatmap_pos[:bbox_num, :], 1, gt_labels.unsqueeze(1), 1)
        cls_heatmap_neg = 1 - cls_heatmap_pos
        cls_heatmap_neg[-2, :] = 0
        lhw_heatmap = torch.zeros([self.max_num_pre_img, 3], dtype=torch.float32)
        lhw_heatmap[:bbox_num, :] = (lhws / self.base_dims[0, gt_labels, :]).log() / self.base_dims[1, gt_labels, :]
        uv_heatmap = torch.zeros([self.max_num_pre_img, 2], dtype=torch.float32)
        uv_heatmap[:bbox_num, :] = center_3ds
        corners_2d_heatmap = torch.zeros([self.max_num_pre_img, 16], dtype=torch.float32)
        corners_2d_heatmap[:bbox_num, :] = corners_2d
        corner_heatmap = torch.zeros([self.max_num_pre_img, 3, 8], dtype=torch.float32)
        corner_heatmap[:bbox_num, :, :] = corners.transpose(1, 2)
        bbox2d_heatmap = torch.zeros([self.max_num_pre_img, 4], dtype=torch.float32)
        bbox2d_heatmap[:bbox_num, :] = torch.from_numpy(gt_bboxes)
        bbox2d_mask = torch.zeros([self.max_num_pre_img, 1], dtype=torch.bool)
        bbox2d_mask[:bbox_num, :] = True
        bbox3d_heatmap = torch.zeros([self.max_num_pre_img, 7], dtype=torch.float32)
        bbox3d_heatmap[:bbox_num, :] = torch.cat((xyzs, lhws, rys.unsqueeze(1)), 1)
        d_heatmap = torch.ones([self.max_num_pre_img, 1], dtype=torch.float32)
        # d_heatmap[:bbox_num, 0] = (depths / cam2img[1, 1] - self.base_depth[0]) / self.base_depth[1]
        d_heatmap[:bbox_num, 0] = depths

        '''
        corner_u = points_cam2img(corners[:, [0, 1, 5, 4], :].view(-1, 3), cam2img).view(-1, 4, 2)[:, :, 0]
        w1w2w3w4s = corner_u - corner_u[:, [1, 2, 3, 0]]
        face_labels = w1w2w3w4s > 4
        w1w2w3w4s = w1w2w3w4s * face_labels
        front_center = torch.mean(corners[:, 4:, :], dim=1, keepdim=False)
        bottom_center = gt_bboxes_3d.bottom_center
        left_center = torch.mean(corners[:, [1, 2, 5, 6], :], dim=1, keepdim=False)
        front_center_2d = points_cam2img(front_center, cam2img)
        bottom_center_2d = points_cam2img(bottom_center, cam2img)
        left_center_2d = points_cam2img(left_center, cam2img)
        '''
        if len(boxes_ignore) > 0:
            if self.train_without_ignore:
                w = boxes_ignore[:, 2] - boxes_ignore[:, 0]
                h = boxes_ignore[:, 3] - boxes_ignore[:, 1]
                max_h_w = np.sqrt(w * h)
                which_preds = np.sum((max_h_w[None, :] > size_distribution[:, None]).astype(np.long), axis=0,
                                     keepdims=True)
                # which_preds = np.array([[0], [1], [2]], dtype=np.int32).repeat(len(boxes_ignore), axis=1)
                for which_pred in which_preds:
                    for i, box in enumerate(boxes_ignore):
                        box = box / down_factor[which_pred[i]]
                        x1 = (max(0, round(box[0])))
                        y1 = (max(0, round(box[1])))
                        x2 = (min(round(box[2]), heat_map_sizes[which_pred[i]][1] - 1))
                        y2 = (min(round(box[3]), heat_map_sizes[which_pred[i]][0] - 1))
                        index_heatmaps[which_pred[i]][0, y1:y2 + 1, x1:x2 + 1] = self.max_num_pre_img - 2
                        if 'gt_labels_ignore' in results.keys():
                            center_heatmap_negs[which_pred[i]][results['gt_labels_ignore'][i], y1:y2 + 1, x1:x2 + 1] = 0
                            # center_heatmap_temps[which_pred[i]][results['gt_labels_ignore'][i], y1:y2 + 1,x1:x2 + 1] = 1
                        else:
                            assert self.train_without_ignore
                            center_heatmap_negs[which_pred[i]][:, y1:y2 + 1, x1:x2 + 1] = 0
                            # center_heatmap_temps[which_pred[i]][:, y1:y2 + 1, x1:x2 + 1] = 1
        if len(gt_bboxes) > 0:
            ws = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            hs = gt_bboxes[:, 3] - gt_bboxes[:, 1]

            max_h_w = np.sqrt(ws * hs)
            which_predss = np.sum((max_h_w[None, :] > size_distribution[:, None]).astype(np.long), axis=0,
                                  keepdims=True)
            # which_predss = np.array([[0], [1], [2]], dtype=np.int32).repeat(len(boxes), axis=1)
            for which_preds in which_predss:
                for i, box in enumerate(gt_bboxes):
                    which_pred = which_preds[i]
                    label = gt_labels[i]
                    box = box / down_factor[which_pred]
                    img_shape_down = img_shape / down_factor[which_pred]
                    x1f, y1f, x2f, y2f = box
                    width = x2f - x1f
                    height = y2f - y1f
                    if self.center_type == '3d':
                        center_x, center_y = center_3ds[i, 0] / down_factor[which_pred], center_3ds[i, 1] / down_factor[
                            which_pred]
                        center_x_int, center_y_int = round(center_x.item()), round(center_y.item())
                    else:
                        center_x, center_y = 0.5 * (x1f + x2f), 0.5 * (y1f + y2f)
                        center_x_int, center_y_int = round(center_x), round(center_y)
                    x1, y1, x2, y2 = round(x1f), round(y1f), round(x2f), round(y2f)
                    if center_y_int < 0 or center_y_int >= heat_map_sizes[which_pred][
                        0] or center_x_int < 0 or center_x_int >= heat_map_sizes[which_pred][1]:
                        center_x_int = max(0, min(center_x_int, heat_map_sizes[which_pred][1] - 1))
                        center_y_int = max(0, min(center_y_int, heat_map_sizes[which_pred][0] - 1))
                    x1 = min(x1, center_x_int)
                    y1 = min(y1, center_y_int)
                    x2 = max(x2, center_x_int)
                    y2 = max(y2, center_y_int)
                    weight = 1
                    x_cood = xs[which_pred][y1:y2 + 1, x1:x2 + 1]
                    y_cood = ys[which_pred][y1:y2 + 1, x1:x2 + 1]
                    center_heatmap_temp = torch.exp(-(((x_cood - center_x_int) / (k_heat * width)) ** 2 + (
                            (y_cood - center_y_int) / (k_heat * height)) ** 2) / 2)
                    mask = center_heatmap_temp >= center_heatmap_temps[which_pred][0, y1:y2 + 1, x1:x2 + 1]
                    center_heatmap_temps[which_pred][0, y1:y2 + 1, x1:x2 + 1][mask] = center_heatmap_temp[mask]
                    if self.iou_heat:
                        x1f, y1f, x2f, y2f = torch.from_numpy(box)
                        width = x2f - x1f
                        height = y2f - y1f
                        iou2 = (torch.minimum(x_cood + k_heat * width / 2, x2f) - torch.maximum(
                            x_cood - k_heat * width / 2, x1f)) * \
                               (torch.minimum(y_cood + k_heat * height / 2, y2f) - torch.maximum(
                                   y_cood - k_heat * height / 2, y1f)) / \
                               (k_heat * width * k_heat * height)
                        neg_weight = torch.clamp_max(2 * (1 - iou2), 1)
                    else:
                        if self.beta == 'inf' or self.free_loss:
                            neg_weight = torch.zeros_like(center_heatmap_temp)
                        else:
                            neg_weight = torch.pow(1 - center_heatmap_temp, self.beta)
                    # neg_weight[neg_weight < 0.5] = 0
                    # neg_weight_debug = neg_weight.numpy()

                    center_heatmap_negs[which_pred][label, y1:y2 + 1, x1:x2 + 1][mask] = neg_weight[mask]
                    if not self.free_loss:
                        center_heatmap_poses[which_pred][:, center_y_int, center_x_int] = 0
                        center_heatmap_poses[which_pred][label, center_y_int, center_x_int] = weight
                    center_heatmap_negs[which_pred][label, center_y_int, center_x_int] = 0
                    index_heatmaps[which_pred][0, y1:y2 + 1, x1:x2 + 1][mask] = i
                    size_masks[which_pred][0, center_y_int, center_x_int] = weight
                    size_heatmaps[which_pred][0, center_y_int, center_x_int] = math.log(width)
                    size_heatmaps[which_pred][1, center_y_int, center_x_int] = math.log(height)
                    size_heatmaps[which_pred][2, center_y_int, center_x_int] = center_x - center_x_int
                    size_heatmaps[which_pred][3, center_y_int, center_x_int] = center_y - center_y_int
        # size_heatmap_debug = size_heatmap.numpy()
        # center_heatmap_pos_debug = center_heatmap_poses[0].numpy()
        # lhw_heatmap_debug = lhw_heatmap.numpy()
        # center_heatmap_neg_debug = center_heatmap_negs[0].numpy()
        # size_mask_debug = size_masks[0].numpy()
        # index_heatmaps_debug = index_heatmaps[0].numpy()

        results['center_heatmap_pos'] = center_heatmap_poses
        results['center_heatmap_neg'] = center_heatmap_negs
        results['size_heatmap'] = size_heatmaps
        results['size_mask'] = size_masks
        results['index_heatmap'] = index_heatmaps

        results['cls_heatmap_pos'] = cls_heatmap_pos
        results['cls_heatmap_neg'] = cls_heatmap_neg
        results['bbox2d_heatmap'] = bbox2d_heatmap
        results['bbox2d_mask'] = bbox2d_mask
        results['bbox3d_heatmap'] = bbox3d_heatmap
        results['lhw_heatmap'] = lhw_heatmap
        results['uv_heatmap'] = uv_heatmap
        results['corners_2d_heatmap'] = corners_2d_heatmap
        results['d_heatmap'] = d_heatmap
        results['corner_heatmap'] = corner_heatmap
        results['alpha_heatmap'] = alpha_heatmap
        results['alpha_4bin_heatmap'] = alpha4bin_heatmap
        return results


@PIPELINES.register_module()
class MakeHeatMap3d():
    def __init__(self, size, label_num, down_factor, kernel_size, size_distribution, base_depth, base_dims,
                 train_without_ignore,
                 train_without_outbound,
                 train_without_small, train_without_truncated, train_without_occluded=3):
        self.size = size
        self.label_num = label_num
        self.down_factor = down_factor
        self.train_without_ignore = train_without_ignore
        self.train_without_outbound = train_without_outbound
        self.train_without_small = train_without_small
        self.train_without_truncated = train_without_truncated
        self.train_without_occluded = train_without_occluded
        self.k_heat = kernel_size
        self.size_distribution = size_distribution
        self.base_depth = base_depth
        self.base_dims = torch.tensor(base_dims)

    def compute_alpha4bin(self, alpha):
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        if alpha > 0.5 * np.pi:
            bin = 3
            alpha2 = np.pi - alpha
        elif alpha > 0:
            bin = 2
            alpha2 = alpha
        elif alpha > -0.5 * np.pi:
            bin = 1
            alpha2 = - alpha
        else:
            bin = 0
            alpha2 = np.pi + alpha
        return alpha2, bin

    def gen_init_tensor(self, down_factor):
        heat_map_size = torch.tensor(self.size) // down_factor
        center_heatmap_pos = torch.zeros([self.label_num, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        center_heatmap_neg = torch.ones([self.label_num, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        face_heatmap = torch.zeros([4, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        h_heatmap = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        d_heatmap = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        ltrb_heatmap = torch.zeros([4, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        size_mask = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        size_heatmap = torch.ones([4, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        lhw_heatmap = torch.zeros([3, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        xyzlhwry_heatmap = torch.zeros([7, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        corner_heatmap = torch.zeros([24, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        offset_heatmap = torch.zeros([2, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        sincos_heatmap = torch.zeros([2, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        alpha_heatmap = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        alpha_4bin_heatmap = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.long)
        x = torch.arange(0, heat_map_size[1])
        x = x.unsqueeze(0).expand([heat_map_size[0], heat_map_size[1]])
        y = torch.arange(0, heat_map_size[0])
        y = y.unsqueeze(1).expand([heat_map_size[0], heat_map_size[1]])
        return heat_map_size, center_heatmap_pos, center_heatmap_neg, face_heatmap, h_heatmap, d_heatmap, size_mask, size_heatmap, lhw_heatmap, offset_heatmap, sincos_heatmap, ltrb_heatmap, alpha_heatmap, alpha_4bin_heatmap, xyzlhwry_heatmap, corner_heatmap, x, y

    def __call__(self, results):
        size_distribution = np.array(self.size_distribution)
        img_shape = np.array(results['img_shape'][:2], dtype=np.float32)
        down_factor = np.array(self.down_factor)
        k_heat = self.k_heat
        heat_map_sizes, center_heatmap_poses, center_heatmap_negs, face_heatmaps, h_heatmaps, d_heatmaps, size_masks, size_heatmaps, lhw_heatmaps, offset_heatmaps, sincos_heatmaps, ltrb_heatmaps, alpha_heatmaps, alpha_4bin_heatmaps, xyzlhwry_heatmaps, corner_heatmaps, xs, ys = multi_apply(
            self.gen_init_tensor, self.down_factor)
        gt_bboxes_3d = results['gt_bboxes_3d']
        boxes_ignore = results['gt_bboxes_ignore']
        labels = results['gt_labels']
        lhws = gt_bboxes_3d.dims
        xyzs = gt_bboxes_3d.gravity_center
        boxes = results['gt_bboxes']
        cam2img = torch.tensor(results['cam2img'])
        K_out = results['K_out']
        depths = results['depths']
        center_2ds = results['centers2d']
        truncateds = results['ann_info']['truncated']
        occludeds = results['ann_info']['occluded']
        # center_2ds = points_cam2img(gt_bboxes_3d.gravity_center, cam2img)
        corners = gt_bboxes_3d.corners
        corner_u = points_cam2img(corners[:, [0, 1, 5, 4], :].view(-1, 3), cam2img).view(-1, 4, 2)[:, :, 0]
        w1w2w3w4s = corner_u - corner_u[:, [1, 2, 3, 0]]
        face_labels = w1w2w3w4s > 4
        w1w2w3w4s = w1w2w3w4s * face_labels
        h2ds = 2 * (points_cam2img(gt_bboxes_3d.bottom_center, cam2img)[:, 1] - center_2ds[:, 1])
        rys = gt_bboxes_3d.tensor[..., 6]
        alphas = rys - torch.atan2(gt_bboxes_3d.tensor[..., 2] + K_out[2], gt_bboxes_3d.tensor[..., 0] + K_out[0])
        '''
        front_center = torch.mean(corners[:, 4:, :], dim=1, keepdim=False)
        bottom_center = gt_bboxes_3d.bottom_center
        left_center = torch.mean(corners[:, [1, 2, 5, 6], :], dim=1, keepdim=False)
        front_center_2d = points_cam2img(front_center, cam2img)
        bottom_center_2d = points_cam2img(bottom_center, cam2img)
        left_center_2d = points_cam2img(left_center, cam2img)
        '''
        if len(boxes_ignore) > 0:
            if self.train_without_ignore:
                w = boxes_ignore[:, 2] - boxes_ignore[:, 0]
                h = boxes_ignore[:, 3] - boxes_ignore[:, 1]
                max_h_w = np.maximum(w, h)
                which_preds = np.sum((max_h_w[None, :] > size_distribution[:, None]).astype(np.long), axis=0,
                                     keepdims=True)
                # which_preds = np.array([[0], [1], [2]], dtype=np.int32).repeat(len(boxes_ignore), axis=1)
                for which_pred in which_preds:
                    for i, box in enumerate(boxes_ignore):
                        box = box / down_factor[which_pred[i]]
                        x1 = int(max(0, (box[0])))
                        y1 = int(max(0, (box[1])))
                        x2 = int(min((box[2]), heat_map_sizes[which_pred[i]][1] - 1))
                        y2 = int(min((box[3]), heat_map_sizes[which_pred[i]][0] - 1))
                        if 'gt_labels_ignore' in results.keys():
                            center_heatmap_negs[which_pred[i]][results['gt_labels_ignore'][i], y1:y2 + 1, x1:x2 + 1] = 0
                        else:
                            assert self.train_without_ignore
                            center_heatmap_negs[which_pred[i]][:, y1:y2 + 1, x1:x2 + 1] = 0
        if len(boxes) > 0:
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            max_h_w = np.maximum(ws, hs)
            which_predss = np.sum((max_h_w[None, :] > size_distribution[:, None]).astype(np.long), axis=0,
                                  keepdims=True)
            # which_predss = np.array([[0], [1], [2]], dtype=np.int32).repeat(len(boxes), axis=1)
            for which_preds in which_predss:
                for box, center_2d, label, lhw, xyz, ry, face_label, w1w2w3w4, h2d, depth, alpha, corner, which_pred, truncated, occluded in zip(
                        boxes, center_2ds, labels, lhws, xyzs, rys, face_labels, w1w2w3w4s, h2ds, depths, alphas,
                        corners, which_preds, truncateds, occludeds):
                    box = box / down_factor[which_pred]
                    w1w2w3w4 = w1w2w3w4 / down_factor[which_pred]
                    center_2d = center_2d / down_factor[which_pred]
                    h2d_down = h2d / down_factor[which_pred]
                    img_shape_down = img_shape / down_factor[which_pred]
                    x1f, y1f, x2f, y2f = box
                    # center_x, center_y = center_2d
                    center_x, center_y = 0.5 * (x1f + x2f), 0.5 * (y1f + y2f)
                    width = x2f - x1f
                    height = y2f - y1f
                    center_x_int, center_y_int = int(center_x), int(center_y)
                    center_x, center_y = center_2d
                    x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
                    if width * down_factor[which_pred] < self.train_without_small[1] or height * down_factor[
                        which_pred] < self.train_without_small[0]:
                        # center_heatmap_negs[which_pred][label, y1:y2 + 1, x1:x2 + 1] = 0
                        continue
                    if center_y_int < 0 or center_y_int >= heat_map_sizes[which_pred][
                        0] or center_x_int < 0 or center_x_int >= heat_map_sizes[which_pred][1]:
                        if self.train_without_outbound:
                            continue
                        else:
                            center_x_int = max(0, min(center_x_int, heat_map_sizes[which_pred][1] - 1))
                            center_y_int = max(0, min(center_y_int, heat_map_sizes[which_pred][0] - 1))
                    x1 = min(x1, center_x_int)
                    y1 = min(y1, center_y_int)
                    x2 = max(x2, center_x_int)
                    y2 = max(y2, center_y_int)
                    '''
                    if self.train_without_outbound:
                        if center_y < y1f or center_y > y2f or center_x < x1f or center_x > x2f:
                            continue
                    else:
                        center_x = min(max(x1f, center_x), x2f)
                        center_y = min(max(y1f, center_y), y2f)
                    center_x_int = int(max(0, min(center_x, heat_map_sizes[which_pred][1] - 1)))
                    center_y_int = int(max(0, min(center_y, heat_map_sizes[which_pred][0] - 1)))
                    x1 = int(max(0, x1f))
                    y1 = int(max(0, y1f))
                    x2 = int(min(x2f, heat_map_sizes[which_pred][1] - 1))
                    y2 = int(min(y2f, heat_map_sizes[which_pred][0] - 1))
                    '''
                    '''
                    center_x_in = min(max(0, center_x), heat_map_sizes[which_pred][1] - 1)
                    center_y_in = min(max(0, center_y), heat_map_sizes[which_pred][0] - 1)
                    x1f = min(x1f, center_x_in)
                    y1f = min(y1f, center_y_in)
                    x2f = max(x2f, center_x_in)
                    y2f = max(y2f, center_y_in)
                    x1 = int(max(0, x1f))
                    y1 = int(max(0, y1f))
                    x2 = int(min(x2f, heat_map_sizes[which_pred][1] - 1))
                    y2 = int(min(y2f, heat_map_sizes[which_pred][0] - 1))
                    if self.train_without_outbound:
                        center_x_int, center_y_int = int(center_x), int(center_y)
                    else:
                        center_x_int, center_y_int = int(center_x_in), int(center_y_in)
                    '''
                    weight = 1
                    center_heatmap_temp = torch.exp(
                        -(((xs[which_pred][y1:y2 + 1, x1:x2 + 1] - center_x_int) / (k_heat * width)) ** 2 + (
                                (ys[which_pred][y1:y2 + 1, x1:x2 + 1] - center_y_int) / (
                                k_heat * height)) ** 2) / 2)
                    center_heatmap_negs[which_pred][label, y1:y2 + 1, x1:x2 + 1] = torch.min(
                        center_heatmap_negs[which_pred][label, y1:y2 + 1, x1:x2 + 1],
                        torch.pow(1 - center_heatmap_temp, 4))
                    if center_y_int > 0 and center_x_int > 0 and center_y_int < heat_map_sizes[which_pred][
                        0] and center_x_int < heat_map_sizes[which_pred][1]:
                        center_heatmap_negs[which_pred][label, center_y_int, center_x_int] = 0
                        center_heatmap_poses[which_pred][:, center_y_int, center_x_int] = 0
                        center_heatmap_poses[which_pred][label, center_y_int, center_x_int] = weight
                        size_masks[which_pred][0, center_y_int, center_x_int] = weight
                        size_heatmaps[which_pred][:, center_y_int, center_x_int] = w1w2w3w4.log()
                        offset_heatmaps[which_pred][0, center_y_int, center_x_int] = center_x - center_x_int
                        offset_heatmaps[which_pred][1, center_y_int, center_x_int] = center_y - center_y_int
                        h_heatmaps[which_pred][0, center_y_int, center_x_int] = h2d_down.log()
                        lhw_heatmaps[which_pred][:, center_y_int, center_x_int] = (lhw / self.base_dims[label, :]).log()
                        xyzlhwry_heatmaps[which_pred][:3, center_y_int, center_x_int] = xyz
                        xyzlhwry_heatmaps[which_pred][3:6, center_y_int, center_x_int] = lhw
                        xyzlhwry_heatmaps[which_pred][6, center_y_int, center_x_int] = ry
                        corner_heatmaps[which_pred][:, center_y_int, center_x_int] = corner.view(-1)
                        face_heatmaps[which_pred][:, center_y_int, center_x_int] = face_label
                        alpha2, bin = self.compute_alpha4bin(alpha)
                        alpha_heatmaps[which_pred][0, center_y_int, center_x_int] = alpha2
                        sincos_heatmaps[which_pred][0, center_y_int, center_x_int] = torch.sin(alpha2)
                        sincos_heatmaps[which_pred][1, center_y_int, center_x_int] = torch.cos(alpha2)
                        alpha_4bin_heatmaps[which_pred][0, center_y_int, center_x_int] = bin
                        # d_heatmaps[which_pred][0, center_y_int, center_x_int] = math.exp(-depth)
                        d_heatmaps[which_pred][0, center_y_int, center_x_int] = float(
                            lhw[1] / h2d - self.base_depth[0]) / self.base_depth[1]
                        ltrb_heatmaps[which_pred][0, center_y_int, center_x_int] = math.log(
                            max(0, center_x_int - x1f) + 1)
                        ltrb_heatmaps[which_pred][1, center_y_int, center_x_int] = math.log(
                            max(0, center_y_int - y1f) + 1)
                        ltrb_heatmaps[which_pred][2, center_y_int, center_x_int] = math.log(
                            max(0, x2f - center_x_int) + 1)
                        ltrb_heatmaps[which_pred][3, center_y_int, center_x_int] = math.log(
                            max(0, y2f - center_y_int) + 1)
        # size_heatmap_debug = size_heatmap.numpy()
        # center_heatmap_pos_debug = center_heatmap_pos.numpy()
        # lhw_heatmap_debug = lhw_heatmap.numpy()
        # center_heatmap_neg_debug = center_heatmap_neg.numpy()
        # size_mask_debug = size_mask.numpy()
        results['center_heatmap_pos'] = center_heatmap_poses
        results['center_heatmap_neg'] = center_heatmap_negs
        results['face_heatmap'] = face_heatmaps
        results['size_heatmap'] = size_heatmaps
        results['lhw_heatmap'] = lhw_heatmaps
        results['size_mask'] = size_masks
        results['offset_heatmap'] = offset_heatmaps
        results['h_heatmap'] = h_heatmaps
        results['d_heatmap'] = d_heatmaps
        results['sincos_heatmap'] = sincos_heatmaps
        results['ltrb_heatmap'] = ltrb_heatmaps
        results['alpha_heatmap'] = alpha_heatmaps
        results['alpha_4bin_heatmap'] = alpha_4bin_heatmaps
        results['corner_heatmap'] = corner_heatmaps
        results['xyz_heatmap'] = [xyzlhwry[:3, :, :] for xyzlhwry in xyzlhwry_heatmaps]
        bbox3d_bk7 = [xyzlhwry.view(7, -1).transpose(0, 1) for xyzlhwry in xyzlhwry_heatmaps]
        results['bbox3d_bk7'] = bbox3d_bk7
        return results


@PIPELINES.register_module()
class RandomResize3d:
    def __init__(self, ratio_range=(0.25, 1.75), img_shape=None, bbox_clip_border=False, backend='cv2'):
        self.backend = backend
        self.ratio_range = ratio_range
        self.img_shape = img_shape
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.img_shape is not None:
            scale, scale_idx = self.random_sample_ratio((self.img_shape[1], self.img_shape[0]),
                                                        self.ratio_range)
        else:
            scale, scale_idx = self.random_sample_ratio((results['img_shape'][1], results['img_shape'][0]),
                                                        self.ratio_range)
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img, w_scale, h_scale = mmcv.imresize(
                results[key],
                results['scale'],
                return_scale=True,
                backend=self.backend)
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            if key in results.keys():
                bboxes = results[key] * np.tile(results['scale_factor'],
                                                results[key].shape[1] // results['scale_factor'].shape[0])
                if self.bbox_clip_border:
                    img_shape = results['img_shape']
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                results[key] = bboxes
        if 'centers2d' in results:
            results['centers2d'] = results['centers2d'] * results['scale_factor'][None, :2]

    def _resize_cam2img(self, results):
        wh_scale = results['scale_factor'][:2]
        results['cam2img'][:2, :] *= wh_scale[:, None]

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_cam2img(results)
        return results


@PIPELINES.register_module()
class ExpandOrCrop3d:
    def __init__(self, size):
        self.size = size

    def crop_box(self, results, left, top, W, H):
        if len(results['gt_bboxes']) > 0:
            box = results['gt_bboxes'].copy()
            box[:, ::2] += left
            box[:, 1::2] += top
            box[:, 0::2] = np.clip(box[:, 0::2], 0, W - 1)
            box[:, 1::2] = np.clip(box[:, 1::2], 0, H - 1)
            after_h = box[:, 3] - box[:, 1]
            after_w = box[:, 2] - box[:, 0]
            after_area = after_w * after_h
            keep_inds = after_area > 1
            object_number = np.sum(keep_inds)
            if object_number == 0:
                return None
            else:
                bias = np.array([[left, top]])
                results['gt_bboxes'] = box[keep_inds, :]
                results['gt_labels'] = results['gt_labels'][keep_inds]
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][keep_inds]
                results['centers2d'] = results['centers2d'][keep_inds, :] + bias
                results['depths'] = results['depths'][keep_inds]
                results['gt_labels_3d'] = results['gt_labels_3d'][keep_inds]
                # results['truncated'] = results['truncated'][keep_inds]
                # results['occluded'] = results['occluded'][keep_inds]
                # results['difficulty'] = results['difficulty'][keep_inds]

        if len(results['gt_bboxes_ignore']) > 0:
            box = results['gt_bboxes_ignore'].copy()
            box[:, ::2] += left
            box[:, 1::2] += top
            box[:, 0::2] = np.clip(box[:, 0::2], 0, W - 1)
            box[:, 1::2] = np.clip(box[:, 1::2], 0, H - 1)
            after_h = box[:, 3] - box[:, 1]
            after_w = box[:, 2] - box[:, 0]
            after_area = after_w * after_h
            keep_inds = after_area > 1
            results['gt_bboxes_ignore'] = box[keep_inds, :]
        return results

    def crop_img(self, image, top, left, bottom, right, H, W):
        image = cv2.copyMakeBorder(image, max(top, 0), max(bottom, 0), max(left, 0), max(right, 0), cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        image = image[max(0, -top):max(0, -top) + H, max(0, -left):max(0, -left) + W, :]
        return image

    def crop_cam2img(self, results, left, top):
        bias = np.array((left, top))
        results['cam2img'][:2, 2] += bias
        results['cam2img'][:2, 3] += (results['cam2img'][2, 3] * bias)
        return results

    def __call__(self, results):
        image = results['img']
        height, width, depth = image.shape
        H, W = self.size
        while True:
            top = random.randint(1 - height, H)
            left = random.randint(1 - width, W)
            # top = random.randint(-16, 16)
            # left = random.randint(-16, 16)
            right = W - width - left
            bottom = H - height - top
            bb = self.crop_box(results, left, top, W, H)
            if bb is not None:
                results = bb
                results['img'] = self.crop_img(image, top, left, bottom, right, H, W)
                results = self.crop_cam2img(results, left, top)
                break
        '''
        #cv2.namedWindow('1', 0)
        #cv2.resizeWindow('1', 1800, 900)
        image2 = image.copy()
        for boxe, lab in zip(results['bbox8d'].copy().astype(np.int32).reshape(-1,4,2), results['facelabel']):
            cv2.polylines(image2, boxe[None,:,:], isClosed=True, color=(int(lab==0) * 255, int(lab==1)*255, int(lab==2)*255), thickness=2)
        cv2.imwrite('1.jpg', image2)
        '''

        '''
        image2 = results['img'].copy()
        for boxe, lab in zip(results['bbox8d'].copy().astype(np.int32).reshape(-1,4,2), results['facelabel']):
            cv2.polylines(image2, boxe[None,:,:], isClosed=True, color=(int(lab==0) * 255, int(lab==1)*255, int(lab==2)*255), thickness=2)
        cv2.imwrite('2.jpg', image2)
        #cv2.waitKey()
        '''
        return results


@PIPELINES.register_module()
class Bbox8dtoXyzxyz:
    def __call__(self, results):
        lhw = results['bbox8d_lhw']
        bbox8d = results['bbox8d']
        H, W, L = np.split(lhw, 3, axis=1)
        K_inv = results['P2_inv']
        K_inv, bias = np.split(K_inv, [3], axis=1)
        fv = results['fv']
        h1h2 = bbox8d[:, 3::4] - bbox8d[:, 1::4]
        d1d2 = H / h1h2 * fv
        x1x2 = bbox8d[:, 2::4] * d1d2
        y1y2 = bbox8d[:, 3::4] * d1d2
        uvduvd = np.stack([x1x2, y1y2, d1d2], axis=2)
        xyzxyz = np.matmul(K_inv[None, None, :, :], uvduvd[:, :, :, None] - bias[None, None, :, :])
        xyzxyz = xyzxyz.reshape([-1, 6])
        results['xyzxyz'] = xyzxyz
        return results


@PIPELINES.register_module()
class Img2Cam:
    def __call__(self, results):
        cam2img = results['cam2img']
        results['img2cam'] = np.linalg.inv(cam2img)
        fu = cam2img[0, 0]
        fv = cam2img[1, 1]
        cu = cam2img[0, 2]
        cv = cam2img[1, 2]
        K_out = np.array(
            [(cam2img[0, 3] - cu * cam2img[2, 3]) / fu, (cam2img[1, 3] - cv * cam2img[2, 3]) / fv, cam2img[2, 3]])
        results['K_out'] = K_out
        return results


@PIPELINES.register_module()
class Init:
    def __call__(self, results):
        bias = np.array([0, 0, 0, 0], dtype=np.float32)
        results['pad_bias'] = bias
        scale_factor = np.array([1, 1, 1, 1], dtype=np.float32)
        results['scale_factor'] = scale_factor
        h, w = results['img_shape'][:2]
        results['xy_max'] = np.array([w, h, w, h], dtype=np.float32)
        results['xy_min'] = np.array([0, 0, 0, 0], dtype=np.float32)
        return results


'''
@PIPELINES.register_module()
class XYMinMax:
    def __call__(self, results):
        h, w = results['img_shape'][:2]
        results['xy_max'] = np.array([w, h, w, h], dtype=np.float32)
        results['xy_min'] = np.array([0, 0, 0, 0], dtype=np.float32)
        return results
'''


@PIPELINES.register_module()
class UnifiedIntrinsics:
    def __init__(self, size=(384, 1280), intrinsics=((721.5377, 0.0, 639.5), (0.0, 721.5377, 172.854), (0.0, 0.0, 1.0)),
                 random_shift=0, random_scale=(1, 1)):
        self.intrinsics = np.array(intrinsics)
        self.size = size
        if random_shift > 0:
            xy_shift = random.randint(-random_shift, random_shift + 1, size=(2,))
            self.intrinsics[0:2, 2] += xy_shift
        if random_scale[0] != 1 or random_scale[1] != 1:
            xy_scale = random.uniform(random_scale[0], random_scale[1], size=(2,))
            self.intrinsics[0, 0] *= xy_scale[0]
            self.intrinsics[1, 1] *= xy_scale[1]

    def _compute_scale(self, results):
        cam2img = results['cam2img']
        w_scale = self.intrinsics[0, 0] / cam2img[0, 0]
        h_scale = self.intrinsics[1, 1] / cam2img[1, 1]
        results['scale_factor'] = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        return results

    def _compute_bias(self, results):
        h, w, _ = results['img_shape']
        cam2img = results['cam2img']
        left = self.intrinsics[0, 2] - cam2img[0, 2]
        top = self.intrinsics[1, 2] - cam2img[1, 2]
        results['pad_bias'] = np.array([left, top, left, top], dtype=np.float32)
        return results

    def _affine_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            A1 = np.zeros([2, 3])
            A1[0, 0] = results['scale_factor'][0]
            A1[1, 1] = results['scale_factor'][1]
            A1[:, 2] = results['pad_bias'][:2]
            img = cv2.warpAffine(img, A1, self.size[::-1], borderValue=0)
            # cv2.imwrite('66.jpg', img)
            results[key] = img
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
        return results

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            if key in results.keys():
                bboxes = results[key] * results['scale_factor'][None, :]
                results[key] = bboxes
        if 'centers2d' in results:
            results['centers2d'] = results['centers2d'] * results['scale_factor'][None, :2]
        return results

    def _resize_cam2img(self, results):
        wh_scale = results['scale_factor'][:2]
        results['cam2img'][:2, :] *= wh_scale[:, None]
        return results

    def _crop_box(self, results):
        bias = results['pad_bias']
        for key in results.get('bbox_fields', []):
            if key in results.keys():
                bbox = results[key] + bias[None, :]
                bbox[:, ::2] = np.clip(bbox[:, ::2], 0, self.size[1])
                bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, self.size[0])
                results[key] = bbox
        if 'centers2d' in results:
            results['centers2d'] = results['centers2d'] + bias[None, :2]
        return results

    def _crop_cam2img(self, results):
        bias = results['pad_bias'][:2]
        results['cam2img'][:2, 2] += bias
        results['cam2img'][:2, 3] += (results['cam2img'][2, 3] * bias)
        return results

    def __call__(self, results):
        results = self._compute_scale(results)
        results = self._resize_bboxes(results)
        results = self._resize_cam2img(results)
        results = self._compute_bias(results)
        results = self._crop_box(results)
        results = self._affine_img(results)
        results = self._crop_cam2img(results)
        return results


@PIPELINES.register_module()
class UnifiedIntrinsics_old:
    def __init__(self, intrinsics=((721.5377, 0.0, 639.5), (0.0, 721.5377, 180.5), (0.0, 0.0, 1.0))):
        self.intrinsics = np.array(intrinsics)

    def _compute_scale(self, results):
        h, w, _ = results['img_shape']
        cam2img = results['cam2img']
        w = round(self.intrinsics[0, 0] * w / cam2img[0, 0])
        h = round(self.intrinsics[1, 1] * h / cam2img[1, 1])
        results['scale'] = (w, h)
        return results

    def _compute_bias(self, results):
        h, w, _ = results['img_shape']
        cam2img = results['cam2img']
        left = round(self.intrinsics[0, 2] - cam2img[0, 2])
        top = round(self.intrinsics[1, 2] - cam2img[1, 2])
        return left, top

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img, w_scale, h_scale = mmcv.imresize(
                results[key],
                results['scale'],
                return_scale=True,
                backend='cv2')
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
        return results

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            if key in results.keys():
                bboxes = results[key] * results['scale_factor'][None, :]
                results[key] = bboxes
        if 'centers2d' in results:
            results['centers2d'] = results['centers2d'] * results['scale_factor'][None, :2]
        return results

    def _resize_cam2img(self, results):
        wh_scale = results['scale_factor'][:2]
        results['cam2img'][:2, :] *= wh_scale[:, None]
        return results

    def _crop_box(self, results, left, top):
        bias = np.array([left, top, left, top], dtype=np.float32)
        results['pad_bias'] = bias
        for key in results.get('bbox_fields', []):
            if key in results.keys():
                results[key] = np.clip(results[key] + bias[None, :], 0, np.inf)
        if 'centers2d' in results:
            results['centers2d'] = results['centers2d'] + bias[None, :2]
        return results

    def _crop_img(self, results, left, top):
        image = results['img']
        h, w, c = results['img_shape']
        image = cv2.copyMakeBorder(image, max(top, 0), 0, max(left, 0), 0, cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        image = image[max(0, -top):, max(0, -left):, :]
        results['img'] = image
        results['img_shape'] = (h + top, w + left, c)
        if h + top > 384:
            print(666)
        if w + left > 1280:
            print(666)
        return results

    def _crop_cam2img(self, results, left, top):
        bias = np.array((left, top))
        results['cam2img'][:2, 2] += bias
        results['cam2img'][:2, 3] += (results['cam2img'][2, 3] * bias)
        return results

    def __call__(self, results):
        results = self._compute_scale(results)
        results = self._resize_img(results)
        results = self._resize_bboxes(results)
        results = self._resize_cam2img(results)
        left, top = self._compute_bias(results)
        results = self._crop_box(results, left, top)
        results = self._crop_img(results, left, top)
        results = self._crop_cam2img(results, left, top)
        return results


@PIPELINES.register_module()
class RandomHSV:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. convert color from BGR to HSV
    2. random saturation
    3. random hue
    4. random value
    5. convert color from HSV to BGR


    Args:
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
        value_range (tuple): range of value.
    """

    def __init__(self, saturation_range=(0.5, 2.0), hue_delta=18, value_range=(0.5, 2.0)):

        self.value_lower, self.value_upper = value_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)
            img[..., 1] = np.minimum(img[..., 1], 1)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # random value
        if random.randint(2):
            img[..., 2] *= random.uniform(self.value_lower, self.value_upper)
            img[..., 2] = np.minimum(img[..., 2], 255)
        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)
        results['img'] = img
        return results
