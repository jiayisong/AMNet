import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class MyBBoxCoder(torch.nn.Module):
    def __init__(self, base_dwdh, base_dxdy, depth_type='h2d', alpha_type='4bin', base_depth=None, base_dims=None,
                 base_dudv=None,
                 base_alpha=None, base_iou3d=None):
        super(MyBBoxCoder, self).__init__()
        assert depth_type in ['h2d', 'h3d/h2d', '1/h2d', 'd*hroi', 'd']
        assert alpha_type in ['hsinhcos', 'sincos', '4bin', 'my4bin', 'sincosv2', 'sincosv3', 'sincosv4', 'sincosv5', 'sincosv6']
        self.depth_type = depth_type
        self.alpha_type = alpha_type
        if base_dims is not None:
            self.register_buffer('base_dims', torch.tensor(base_dims, dtype=torch.float32))
        if base_depth is not None:
            self.register_buffer('base_depth', torch.tensor(base_depth, dtype=torch.float32))
        if base_dudv is not None:
            self.register_buffer('base_dudv', torch.tensor(base_dudv, dtype=torch.float32))
        if base_dudv is not None:
            self.register_buffer('base_alpha', torch.tensor(base_alpha, dtype=torch.float32))
        if base_iou3d is not None:
            self.register_buffer('base_iou3d', torch.tensor(base_iou3d, dtype=torch.float32))
        self.register_buffer('base_dwdh', torch.tensor(base_dwdh, dtype=torch.float32))
        self.register_buffer('base_dxdy', torch.tensor(base_dxdy, dtype=torch.float32))
        self.register_buffer('cor',
                             torch.tensor([[-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5],
                                           [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                           [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]], dtype=torch.float32))
        '''
        self.register_buffer('cor',
                             torch.tensor([[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
                                           [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]],
                                          dtype=torch.float32))
        '''
        self.register_buffer('constant_neg1_1', torch.tensor([[-1, 1]], dtype=torch.float32))
        # self.register_buffer('constant_1_neg1_1_neg1', torch.tensor([[1, -1, 1, -1]], dtype=torch.float32))
        # self.register_buffer('constant_0_pi_pi_2pi', torch.tensor([[0, np.pi, -np.pi, 2*np.pi]], dtype=torch.float32))
        # self.register_buffer('constant_1_neg1', torch.tensor([[[[1]], [[-1]]]], dtype=torch.float32))
        self.register_buffer('constant_1_0', torch.tensor([1, 0], dtype=torch.long))
        self.register_buffer('constant_2_0_1', torch.tensor([2, 0, 1], dtype=torch.long))
        self.register_buffer('constant_0', torch.tensor([[[0]]], dtype=torch.float32))
        self.register_buffer('constant_1', torch.tensor([[[1]]], dtype=torch.float32))

    def decode_label(self, cls_pred):
        score, label = torch.max(cls_pred, dim=1, keepdim=False)
        return score, label

    def encode_uv(self, rois, uv):
        '''

        Args:
            rois: [N, 5]
            uv: [N, 2] or [N, 18]

        Returns:

        '''
        xy_min_rois = rois[:, 1:3]
        xy_max_rois = rois[:, 3:5]
        center_rois = 0.5 * (xy_min_rois + xy_max_rois)
        wh_rois = (xy_max_rois - xy_min_rois).clamp_min(8)
        # wh_rois = 40
        N, C = uv.shape
        if C == 2:
            dudv = (uv - center_rois) / wh_rois
        else:
            uv = uv.view(N, C // 2, 2)
            center_rois = center_rois.unsqueeze(1)
            wh_rois = wh_rois.unsqueeze(1)
            dudv = (uv - center_rois) / wh_rois
            dudv = dudv.view(N, C)
        dudv = (dudv - self.base_dudv[0]) / self.base_dudv[1]
        return dudv

    def decode_uv(self, rois, dudv):
        if dudv.shape[1] > 2:
            dudv = dudv[:, :2]
        dudv = dudv * self.base_dudv[1] + self.base_dudv[0]
        xy_min_rois = rois[:, 1:3]
        xy_max_rois = rois[:, 3:5]
        center_rois = 0.5 * (xy_min_rois + xy_max_rois)
        wh_rois = (xy_max_rois - xy_min_rois).clamp_min(8)
        # wh_rois = 40
        uv = center_rois + dudv * wh_rois
        return uv

    def encode_bbox2d(self, rois, bbox2d):
        xy_min, xy_max = torch.split(bbox2d, 2, dim=1)
        xy_min_rois = rois[:, 1:3]
        xy_max_rois = rois[:, 3:5]
        center_rois = 0.5 * (xy_min_rois + xy_max_rois)
        center = 0.5 * (xy_min + xy_max)
        wh_rois = (xy_max_rois - xy_min_rois).clamp_min(8)
        wh = (xy_max - xy_min).clamp_min(8)
        dxdy = (center - center_rois) / wh_rois
        dwdh = (wh / wh_rois).log()
        dxdy = (dxdy - self.base_dxdy[0]) / self.base_dxdy[1]
        dwdh = (dwdh - self.base_dwdh[0]) / self.base_dwdh[1]
        # delta = torch.cat((dxdy, dwdh), dim=1)
        return dxdy, dwdh

    def decode_bbox2d(self, rois, dxdy, dwdh):
        # dxdy, dwdh = torch.split(delta, 2, dim=1)
        dxdy = dxdy * self.base_dxdy[1] + self.base_dxdy[0]
        dwdh = dwdh * self.base_dwdh[1] + self.base_dwdh[0]
        xy_min_rois = rois[:, 1:3]
        xy_max_rois = rois[:, 3:5]
        center_rois = 0.5 * (xy_min_rois + xy_max_rois)
        wh_rois = (xy_max_rois - xy_min_rois).clamp_min(8)
        center = dxdy * wh_rois + center_rois
        wh_half = dwdh.exp() * wh_rois * 0.5
        xy_min = center - wh_half
        xy_max = center + wh_half
        bbox2d = torch.cat((xy_min, xy_max), dim=1)
        return bbox2d

    def decode_lhw(self, lhw_pred, labels):
        '''
        Args:
            lhw_pred:[b,3]
            labels:[b,]
        Returns:
            lhw_pred:[b,3]
        '''
        batch_size, _ = lhw_pred.size()
        lhw_pred = lhw_pred * self.base_dims[1, labels, :]
        lhw_pred = lhw_pred.exp() * self.base_dims[0, labels, :]
        lhw_pred = lhw_pred.clamp_min(1e-1)
        return lhw_pred

    # @profile
    def decode_ry(self, alpha_4bin_pred, alpha_pred, xyz, K_out):
        '''
        Args:
            alpha_4bin_pred: [b,4] or none
            alpha_pred:  [b,4] or[b,2]or[b,1]
            xyz: [b,3]
            K_out: [b,3]or[1,3]
        Returns:
            ry:[b,1]
        '''
        if self.alpha_type == '4bin':
            _, bin = torch.max(alpha_4bin_pred, dim=1, keepdim=True)
            # alpha = torch.gather(alpha_pred, 1, bin)
            alpha = torch.sum(alpha_pred * F.one_hot(bin.squeeze(1), 4), dim=1, keepdim=True)
            alpha = alpha * self.base_alpha[1] + self.base_alpha[0]
            alpha = np.pi / 4 + bin.float() * np.pi / 2 + alpha
            # alpha = np.pi / 4 + bin.float() * np.pi / 2 + (bin % 2 == 0) * alpha - (bin % 2 == 1) * alpha
        elif self.alpha_type == 'my4bin':
            _, bin = torch.max(alpha_4bin_pred, dim=1, keepdim=True)
            alpha = alpha_pred * self.base_alpha[1] + self.base_alpha[0]
            alpha = np.pi / 4 + bin.float() * np.pi / 2 + (bin % 2 == 0) * alpha - (bin % 2 == 1) * alpha
        elif self.alpha_type == 'sincos':
            alpha_pred = alpha_pred * self.base_alpha[1] + self.base_alpha[0]
            alpha = torch.atan2(alpha_pred[:, 0], alpha_pred[:, 1]).unsqueeze(1)
        elif self.alpha_type == 'sincosv2':
            alpha = alpha_pred
        elif self.alpha_type == 'sincosv3':
            alpha = torch.atan2(alpha_pred[:, 0] - alpha_pred[:, 2], alpha_pred[:, 1] - alpha_pred[:, 3]).unsqueeze(1)
        elif self.alpha_type == 'sincosv4':
            alpha_pred = alpha_pred * 2 - 1
            alpha = torch.atan2(alpha_pred[:, 0], alpha_pred[:, 1]).unsqueeze(1)
        elif self.alpha_type == 'sincosv5':
            alpha = torch.atan2(alpha_pred[:, 0], alpha_pred[:, 1]).unsqueeze(1)
        elif self.alpha_type == 'sincosv6':
            alpha = torch.atan2(alpha_pred[:, 0], alpha_pred[:, 1]).unsqueeze(1)
        else:
            alpha_pred = alpha_pred * self.base_alpha[1] + self.base_alpha[0]
            alpha_pred = torch.clamp(alpha_pred, -np.pi / 2, np.pi / 2)
            sin, cos = torch.chunk(alpha_pred, 2, 1)
            sign_sin = torch.sign(sin)
            sign_cos = torch.sign(cos)
            alpha = (sign_sin * (np.pi - cos) + sign_cos * sin - sign_sin * sign_cos * np.pi / 2) / 2
        ry = (torch.atan2(xyz[:, 2] + K_out[:, 2], xyz[:, 0] + K_out[:, 0]).unsqueeze(1) + alpha)
        # ry = alpha.unsqueeze(1)
        ry = (ry + np.pi) % (2 * np.pi) - np.pi
        return ry

    def decode_xyz(self, uv, d, img2cam):
        '''
        Args:
            uv: [b,2]
            d:  [b,1]
            img2cam:[b,4,4] or [1,4,4]
        Returns:
            xyz: [b,3]
        '''
        centers2d_img = torch.cat((uv * d, d, self.constant_1.view(1, 1).expand_as(d)), dim=1).unsqueeze(2)  # [b,4,1]
        xyz = torch.matmul(img2cam, centers2d_img)
        xyz = xyz[:, :3, 0]
        return xyz

    def decode_d(self, rois, d_pred, lhw, cam2img):
        '''

        Args:
            d_pred: [b,1]
            cam2img: [1,4,4] or [b,4,4]
        Returns:
            d: [b,1]
        '''
        h = rois[:, 4] - rois[:, 2]
        if self.depth_type == 'h3d/h2d':
            d = cam2img[:, 1, 1].unsqueeze(1) * (d_pred * self.base_depth[1] + self.base_depth[0]) / h.unsqueeze(1)
        elif self.depth_type == 'h2d':
            h3d = lhw[:, 1].unsqueeze(1)
            d = h3d * cam2img[:, 1, 1].unsqueeze(1) / (
                    (d_pred * self.base_depth[1] + self.base_depth[0]) * h.unsqueeze(1))
        elif self.depth_type == '1/h2d':
            h3d = lhw[:, 1].unsqueeze(1)
            d = h3d * cam2img[:, 1, 1].unsqueeze(1) * (d_pred * self.base_depth[1] + self.base_depth[0]) / h.unsqueeze(
                1)
        elif self.depth_type == 'd*hroi':
            d = (d_pred * self.base_depth[1] + self.base_depth[0]) / h.unsqueeze(1)
        elif self.depth_type == 'd':
            d = (d_pred * self.base_depth[1] + self.base_depth[0])
        else:
            raise NotImplementedError
        return d

    def encode_d(self, rois, d, lhw, cam2img):
        '''

        Args:
            d_pred: [b,1]
            cam2img: [1,4,4] or [b,4,4]
        Returns:
            d: [b,1]
        '''
        h = rois[:, 4] - rois[:, 2]
        if self.depth_type == 'h3d/h2d':
            d_pred = (h.unsqueeze(1) * d / cam2img[:, 1, 1].unsqueeze(1) - self.base_depth[0]) / self.base_depth[1]
        elif self.depth_type == 'h2d':
            h3d = lhw[:, 1].unsqueeze(1)
            d_pred = (h3d * cam2img[:, 1, 1].unsqueeze(1) / (d * h.unsqueeze(1)) - self.base_depth[0]) / \
                     self.base_depth[1]
        elif self.depth_type == '1/h2d':
            h3d = lhw[:, 1].unsqueeze(1)
            d_pred = (d * h.unsqueeze(1) / (h3d * cam2img[:, 1, 1].unsqueeze(1)) - self.base_depth[0]) / \
                     self.base_depth[1]
        elif self.depth_type == 'd*hroi':
            d_pred = (h.unsqueeze(1) * d - self.base_depth[0]) / self.base_depth[1]
        elif self.depth_type == 'd':
            d_pred = (d - self.base_depth[0]) / self.base_depth[1]
        else:
            raise NotImplementedError
        return d_pred

    def decode_corner(self, xyz, lhw, ry):
        '''
        Args:
            xyz: [b,3]
            lhw: [b,3]
            ry: [b,1]
        Returns:
            corners: [b, 3, 8]
        '''
        b = xyz.shape[0]
        cos_ry = torch.cos(ry)
        sin_ry = torch.sin(ry)
        zero = self.constant_0.view(1, 1).expand(b, 1)
        one = self.constant_1.view(1, 1).expand(b, 1)
        rot = torch.cat((cos_ry, zero, sin_ry, zero, one, zero, -sin_ry, zero, cos_ry), dim=1)
        rot = rot.view([b, 3, 3])
        temp2 = lhw.unsqueeze(2) * self.cor
        # temp3 = torch.einsum("ijk,ikn->inj", rot, temp2)
        temp3 = torch.bmm(rot, temp2)
        temp4 = temp3 + xyz.unsqueeze(2)
        corners = temp4
        return corners

    def decode_iou3d(self, iou3d):
        '''
        Args:
            iou3d: [b,1]

        '''
        iou3d = iou3d * self.base_iou3d[1] + self.base_iou3d[0]
        iou3d = iou3d.clamp(0, 1)
        return iou3d

    def encode_iou3d(self, iou3d):
        '''
        Args:
            iou3d: [b,1]

        '''
        iou3d = (iou3d - self.base_iou3d[0]) / self.base_iou3d[1]
        return iou3d
