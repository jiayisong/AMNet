import torch
from torch.nn import functional as F
from mmdet3d.ops.iou3d.iou3d_utils import nms_3d_gpu, nms_bev_gpu, boxes_iou_3d_gpu
from mmdet.core import multi_apply
from mmdet.core.bbox.builder import build_bbox_coder
from mmdet.models.builder import HEADS
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from .anchor_free_mono3d_head import AnchorFreeMono3DHead
from ..losses.gaussian_varifocal_loss import GaussianVariFocalLoss
from mmcv.ops import grid_sample_deterministic


@HEADS.register_module()
class SMOKEMono3DHead(AnchorFreeMono3DHead):
    r"""Anchor-free head used in `SMOKE <https://arxiv.org/abs/2002.10111>`_

    .. code-block:: none

                /-----> 3*3 conv -----> 1*1 conv -----> cls
        feature
                \-----> 3*3 conv -----> 1*1 conv -----> reg

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        dim_channel (list[int]): indices of dimension offset preds in
            regression heatmap channels.
        ori_channel (list[int]): indices of orientation offset pred in
            regression heatmap channels.
        bbox_coder (:obj:`CameraInstance3DBoxes`): Bbox coder
            for encoding and decoding boxes.
        loss_cls (dict, optional): Config of classification loss.
            Default: loss_cls=dict(type='GaussionFocalLoss', loss_weight=1.0).
        loss_bbox (dict, optional): Config of localization loss.
            Default: loss_bbox=dict(type='L1Loss', loss_weight=10.0).
        loss_dir (dict, optional): Config of direction classification loss.
            In SMOKE, Default: None.
        loss_attr (dict, optional): Config of attribute classification loss.
            In SMOKE, Default: None.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict): Initialization config dict. Default: None.
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 dim_channel,
                 ori_channel,
                 bbox_coder,
                 group_reg_loss_weights,
                 loss_cls=dict(type='GaussionFocalLoss', loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.1),
                 loss_dir=None,
                 loss_attr=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.dim_channel = dim_channel
        self.ori_channel = ori_channel
        self.group_reg_loss_weights = group_reg_loss_weights
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward features of a single scale level.

        Args:
            x (Tensor): Input feature map.

        Returns:
            tuple: Scores for each class, bbox of input feature maps.
        """
        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat = \
            super().forward_single(x)
        cls_score = cls_score.sigmoid()  # turn to 0-1
        # cls_score = cls_score.clamp(min=1e-4, max=1 - 1e-4)
        # (N, C, H, W)
        vector_ori = bbox_pred[:, self.ori_channel, ...]
        bbox_pred[:, self.ori_channel, ...] = F.normalize(vector_ori)

        return cls_score, bbox_pred

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cam2img, img2cam, K_out, normal_uv, rescale=None):
        """Generate bboxes from bbox head predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
            bbox_preds (list[Tensor]): Box regression for each scale.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[tuple[:obj:`CameraInstance3DBoxes`, Tensor, Tensor, None]]:
                Each item in result_list is 4-tuple.
        """
        assert len(cls_scores) == len(bbox_preds) == 1
        batch_bboxes, batch_scores, batch_topk_labels = self.decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            img_metas,
            img2cam[0], K_out[0],
            cam2img[0], normal_uv[0],
            topk=self.test_cfg.max_per_img,
            kernel=self.test_cfg.local_maximum_kernel)

        result_list = []
        for img_id in range(len(img_metas)):
            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]

            keep_idx = scores > self.test_cfg.min_score
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            bboxes, scores, labels, keep2 = self.box3d_nms(bboxes, scores, labels)
            bboxes = img_metas[img_id]['box_type_3d'](
                bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
            attrs = None
            result_list.append((bboxes, scores, labels, attrs))

        return result_list

    def decode_heatmap(self,
                       cls_score,
                       reg_pred,
                       img_metas,
                       img2cam, K_out,
                       cam2img, normal_uv,
                       topk=100,
                       kernel=3):
        """Transform outputs into detections raw bbox predictions.

        Args:
            class_score (Tensor): Center predict heatmap,
                shape (B, num_classes, H, W).
            reg_pred (Tensor): Box regression map.
                shape (B, channel, H , W).
            img_metas (List[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cam2imgs (Tensor): Camera intrinsic matrixs.
                shape (B, 4, 4)
            trans_mats (Tensor): Transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
            topk (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of SMOKEHead, containing
               the following Tensors:
              - batch_bboxes (Tensor): Coords of each 3D box.
                    shape (B, k, 7)
              - batch_scores (Tensor): Scores of each 3D box.
                    shape (B, k)
              - batch_topk_labels (Tensor): Categories of each 3D box.
                    shape (B, k)
        """
        bs, _, feat_h, feat_w = cls_score.shape
        if kernel > 1:
            cls_score = get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            cls_score, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets
        #reg_pred, ab, ground_height = torch.split(reg_pred, (8, 2, 1), dim=1)
        regression = transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, 8)
        #h2d, regression = torch.split(regression, (1, 7), dim=1)
        points = torch.cat([topk_xs.view(-1, 1), topk_ys.view(-1, 1).float()], dim=1)
        #centers2d = self.bbox_coder._decode_offset(regression[:, :2], points, False)
        #centers2d[:, 1] = centers2d[:, 1] - (h2d ** 3).squeeze(1) * 10
        #depth = self.get_ground_depth(ab, cam2img, ground_height, normal_uv, centers2d.view(-1, topk, 2), None)
        #regression = torch.cat((depth, regression), dim=1)
        locations, dimensions, orientations = self.bbox_coder.decode(
            regression, points, batch_topk_labels, img2cam, K_out)

        batch_bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        batch_bboxes = batch_bboxes.view(bs, -1, self.bbox_code_size)
        return batch_bboxes, batch_scores, batch_topk_labels

    def get_ground_depth(self, ab_feature, cam2img, ground_height, normal_uv, bottom_center, indice_mask):
        '''

        Args:
            ab_feature: [b,2,h,w]
            cam2img:  [b,4,4]
            ground_height: [b,1,h,w]
            normal_uv: [b,2,h,w]
            bottom_center: [b,N,2]
            indice_mask: [b*N, ]
        Returns:
            ground_depth_posi: [n, 1] or [b*N, 1]
        '''
        # camera_center = cam2img[:, :2, 2:3] / self.bbox_coder.down_ratio
        # fu = cam2img[:, 0, 0]
        # fv = cam2img[:, 1, 1]
        # ab = grid_sample_deterministic(ab_feature, camera_center)  # [b,2,1]
        # ab = ab_feature.mean(dim=(2, 3), keepdim=False)
        # ab = torch.tanh(1 * ab[:, :])
        # ab = ab[:, :, 0]
        # with torch.no_grad():
        #    b_nograd = b
        # b = ab[:, 1] * 70 / fv
        # a = ab[:, 0] * 0.1 * fu / fv
        ground_height = 1.65  # ground_height + 1.65
        temp = normal_uv[:, 1, :, :]  # - a.view(-1, 1, 1) * normal_uv[:, 0, :, :] - b.view(-1, 1, 1)
        temp = temp.clamp_min(0.02).unsqueeze(1)  # [b,1,h,w]
        # with torch.no_grad():
        #       temp2 = ground_height / temp - ground_height
        # ground_depth = ground_height + temp2
        ground_depth = ground_height / temp
        # ground_depth_debug = ground_depth.detach().cpu().numpy()
        # temp_debug = temp.detach().cpu().numpy()
        # bottom_center_debug = bottom_center.detach().cpu().numpy()
        ground_depth_posi = grid_sample_deterministic(ground_depth, bottom_center.transpose(1, 2).contiguous()).view(
            -1, 1)  # [b,N]
        # ground_depth_posi_debug = ground_depth_posi.detach().cpu().numpy()
        if indice_mask is not None:
            ground_depth_posi = ground_depth_posi[indice_mask]
        return ground_depth_posi

    def get_predictions(self, labels3d, centers2d, gt_locations, gt_dimensions, cam2img, z_indice, normal_uv,
                        gt_h2d, gt_orientations, gt_depths, gt_offsets, indices, img_metas, img2cam, K_out, pred_reg,
                        pred_cls):
        """Prepare predictions for computing loss.

        Args:
            labels3d (Tensor): Labels of each 3D box.
                shape (B, max_objs, )
            centers2d (Tensor): Coords of each projected 3D box
                center on image. shape (B * max_objs, 2)
            gt_locations (Tensor): Coords of each 3D box's location.
                shape (B * max_objs, 3)
            gt_dimensions (Tensor): Dimensions of each 3D box.
                shape (N, 3)
            gt_orientations (Tensor): Orientation(yaw) of each 3D box.
                shape (N, 1)
            indices (Tensor): Indices of the existence of the 3D box.
                shape (B * max_objs, )
            img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            pre_reg (Tensor): Box regression map.
                shape (B, channel, H , W).

        Returns:
            dict: the dict has components below:
            - bbox3d_yaws (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred orientations.
            - bbox3d_dims (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred dimensions.
            - bbox3d_locs (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred locations.
        """
        batch = pred_reg.shape[0]
        w = pred_reg.shape[3]

        centers2d_inds = centers2d[:, 1] * w + centers2d[:, 0]
        centers2d_inds = centers2d_inds.view(batch, -1)

        # ab = pred_reg[:, 8:10, :, :]
        # ground_height = pred_reg[:, 10:, :, :]

        # pred_reg = pred_reg[:, :8, :, :]
        pred_regression = transpose_and_gather_feat(pred_reg, centers2d_inds)
        pred_regression_pois = pred_regression.view(-1, 8)
        # h2d, pred_regression_pois = torch.split(pred_regression_pois, [1, 7], dim=1)

        pred_class = transpose_and_gather_feat(pred_cls, centers2d_inds)
        pred_class_pois = pred_class.view(-1, self.num_classes)
        pred_class_pois = torch.gather(pred_class_pois, dim=1, index=labels3d.view(-1, 1)).view(-1)

        # bottom_centers2d = self.bbox_coder._decode_offset(pred_regression_pois[:, 1:3], centers2d.detach(), False)
        # bottom_centers2d[:, 1] = bottom_centers2d[:, 1] - (h2d**3).squeeze(1) * 10

        # z_indice = bottom_centers2d.view(batch, -1, 2)

        # ground_depth = self.get_ground_depth(ab, cam2img, ground_height, normal_uv, z_indice, None)

        # pred_regression_pois = torch.cat((ground_depth, pred_regression_pois), dim=1)

        locations, dimensions, orientations = self.bbox_coder.decode(
            pred_regression_pois, centers2d, labels3d, img2cam, K_out, gt_locations, gt_offsets, gt_depths)

        depth_locations, centers2d_locations, locations = locations
        depth_locations, centers2d_locations, locations, dimensions, orientations = \
            depth_locations[indices], centers2d_locations[indices], locations[indices], dimensions[indices], \
            orientations[indices]

        # h2d = h2d[indices, 0]
        gt_locations = gt_locations[indices]
        pred_class_pois = pred_class_pois[indices]
        assert len(centers2d_locations) == len(gt_locations)
        assert len(depth_locations) == len(gt_locations)
        assert len(dimensions) == len(gt_dimensions)
        assert len(orientations) == len(gt_orientations)
        bbox3d_yaws = self.bbox_coder.encode(gt_locations, gt_dimensions, orientations, img_metas, (0.5, 0.5, 0.5))
        bbox3d_dims = self.bbox_coder.encode(gt_locations, dimensions, gt_orientations, img_metas, (0.5, 0.5, 0.5))
        bbox3d_depths = self.bbox_coder.encode(depth_locations, gt_dimensions, gt_orientations, img_metas,
                                               (0.5, 0.5, 0.5))
        bbox3d_offsets = self.bbox_coder.encode(centers2d_locations, gt_dimensions, gt_orientations, img_metas,
                                                (0.5, 0.5, 0.5))
        with torch.no_grad():
            bbox3d_pred = torch.cat((locations, dimensions, orientations), dim=1)
            gt_bbox3d = torch.cat((gt_locations, gt_dimensions, gt_orientations), dim=1)
            IOU3D = boxes_iou_3d_gpu(bbox3d_pred, gt_bbox3d, aligned=True)

        pred_bboxes = dict(ori=bbox3d_yaws, dim=bbox3d_dims, depth=bbox3d_depths, offset=bbox3d_offsets, iou=IOU3D,
                           #   h2d=h2d
                           )

        return pred_bboxes, pred_class_pois

    def get_targets(self, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,
                    centers2d, feat_shape, img_shape, img_metas):
        """Get training targets for batch images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gt,).
            gt_bboxes_3d (list[:obj:`CameraInstance3DBoxes`]): 3D Ground
                truth bboxes of each image,
                shape (num_gt, bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D Ground truth labels of each
                box, shape (num_gt,).
            centers2d (list[Tensor]): Projected 3D centers onto 2D image,
                shape (num_gt, 2).
            feat_shape (tuple[int]): Feature map shape with value,
                shape (B, _, H, W).
            img_shape (tuple[int]): Image shape in [h, w] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple[Tensor, dict]: The Tensor value is the targets of
                center heatmap, the dict has components below:
              - gt_centers2d (Tensor): Coords of each projected 3D box
                    center on image. shape (B * max_objs, 2)
              - gt_labels3d (Tensor): Labels of each 3D box.
                    shape (B, max_objs, )
              - indices (Tensor): Indices of the existence of the 3D box.
                    shape (B * max_objs, )
              - affine_indices (Tensor): Indices of the affine of the 3D box.
                    shape (N, )
              - gt_locs (Tensor): Coords of each 3D box's location.
                    shape (N, 3)
              - gt_dims (Tensor): Dimensions of each 3D box.
                    shape (N, 3)
              - gt_yaws (Tensor): Orientation(yaw) of each 3D box.
                    shape (N, 1)
              - gt_cors (Tensor): Coords of the corners of each 3D box.
                    shape (N, 8, 3)
        """

        reg_mask = torch.stack([
            gt_bboxes[0].new_tensor(
                not img_meta['affine_aug'], dtype=torch.bool)
            for img_meta in img_metas
        ])

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)  # 1/4
        height_ratio = float(feat_h / img_h)  # 1/4

        assert width_ratio == height_ratio

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])

        gt_centers2d = centers2d.copy()

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            # project centers2d from input image to feat map
            gt_center2d = gt_centers2d[batch_id] * width_ratio

            for j, center in enumerate(gt_center2d):
                center_x_int, center_y_int = center.int()
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.7)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [center_x_int, center_y_int], radius)

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        num_ctrs = [center2d.shape[0] for center2d in centers2d]
        max_objs = max(num_ctrs)

        reg_inds = torch.cat(
            [reg_mask[i].repeat(num_ctrs[i]) for i in range(bs)])

        inds = torch.zeros((bs, max_objs),
                           dtype=torch.bool).to(centers2d[0].device)

        # put gt 3d bboxes to gpu
        gt_bboxes_3d = [
            gt_bbox_3d.to(centers2d[0].device) for gt_bbox_3d in gt_bboxes_3d
        ]

        batch_centers2d = centers2d[0].new_zeros((bs, max_objs, 2))
        batch_labels_3d = gt_labels_3d[0].new_zeros((bs, max_objs))
        batch_gt_locations = \
            gt_bboxes_3d[0].tensor.new_zeros((bs, max_objs, 3))
        for i in range(bs):
            inds[i, :num_ctrs[i]] = 1
            batch_centers2d[i, :num_ctrs[i]] = centers2d[i]
            batch_labels_3d[i, :num_ctrs[i]] = gt_labels_3d[i]
            batch_gt_locations[i, :num_ctrs[i]] = \
                gt_bboxes_3d[i].tensor[:, :3]

        inds = inds.flatten()
        batch_centers2d = batch_centers2d.view(-1, 2) * width_ratio
        batch_gt_locations = batch_gt_locations.view(-1, 3)

        # filter the empty image, without gt_bboxes_3d
        gt_bboxes_3d = [
            gt_bbox_3d for gt_bbox_3d in gt_bboxes_3d
            if gt_bbox_3d.tensor.shape[0] > 0
        ]

        gt_dimensions = torch.cat(
            [gt_bbox_3d.tensor[:, 3:6] for gt_bbox_3d in gt_bboxes_3d])
        gt_orientations = torch.cat([
            gt_bbox_3d.tensor[:, 6].unsqueeze(-1)
            for gt_bbox_3d in gt_bboxes_3d
        ])
        gt_corners = torch.cat(
            [gt_bbox_3d.corners for gt_bbox_3d in gt_bboxes_3d])

        target_labels = dict(
            gt_centers2d=batch_centers2d.long(),
            gt_labels3d=batch_labels_3d,
            indices=inds,
            reg_indices=reg_inds,
            gt_locs=batch_gt_locations,
            gt_dims=gt_dimensions,
            gt_yaws=gt_orientations,
            gt_cors=gt_corners)

        return center_heatmap_target, avg_factor, target_labels

    def loss(self,
             cls_scores,
             bbox_preds, img_metas,
             center_heatmap_target, gt_centers2d, gt_labels3d, indices, reg_indices,
             gt_locs, gt_dims, gt_yaws, gt_cors, gt_depths, gt_offsets, img2cam, K_out,
             cam2img, z_indice, normal_uv, gt_h2d):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
                shape (num_gt, 4).
            bbox_preds (list[Tensor]): Box dims is a 4D-tensor, the channel
                number is bbox_code_size.
                shape (B, 7, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image.
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
                shape (num_gts, ).
            gt_bboxes_3d (list[:obj:`CameraInstance3DBoxes`]): 3D boxes ground
                truth. it is the flipped gt_bboxes
            gt_labels_3d (list[Tensor]): Same as gt_labels.
            centers2d (list[Tensor]): 2D centers on the image.
                shape (num_gts, 2).
            depths (list[Tensor]): Depth ground truth.
                shape (num_gts, ).
            attr_labels (list[Tensor]): Attributes indices of each box.
                In kitti it's None.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == 1

        center2d_heatmap = cls_scores[0]
        pred_reg = bbox_preds[0]

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        gt_centers2d = gt_centers2d.view(-1, 2)
        indices = indices.view(-1)
        reg_inds = torch.cat(reg_indices, dim=0)
        gt_locs = gt_locs.view(-1, 3)
        gt_depths = gt_depths.view(-1)
        gt_offsets = gt_offsets.view(-1, 2)
        gt_dims = torch.cat(gt_dims, dim=0)
        gt_yaws = torch.cat(gt_yaws, dim=0)
        gt_cors = torch.cat(gt_cors, dim=0)
        gt_h2d = torch.cat(gt_h2d, dim=0)
        pred_bboxes, pred_cls = self.get_predictions(
            labels3d=gt_labels3d, centers2d=gt_centers2d, gt_locations=gt_locs, gt_dimensions=gt_dims,
            gt_orientations=gt_yaws, gt_depths=gt_depths, gt_offsets=gt_offsets, indices=indices,
            img_metas=img_metas, img2cam=img2cam, cam2img=cam2img, z_indice=z_indice, normal_uv=normal_uv,
            gt_h2d=gt_h2d, K_out=K_out, pred_reg=pred_reg, pred_cls=center2d_heatmap)

        loss_bbox_depths = self.group_reg_loss_weights[0] * self.loss_bbox(
            pred_bboxes['depth'].corners[reg_inds, ...],
            gt_cors[reg_inds, ...], avg_factor=avg_factor)
        loss_bbox_oris = self.group_reg_loss_weights[3] * self.loss_bbox(
            pred_bboxes['ori'].corners[reg_inds, ...],
            gt_cors[reg_inds, ...], avg_factor=avg_factor)
        loss_bbox_dims = self.group_reg_loss_weights[2] * self.loss_bbox(
            pred_bboxes['dim'].corners[reg_inds, ...],
            gt_cors[reg_inds, ...], avg_factor=avg_factor)
        loss_bbox_offsets = self.group_reg_loss_weights[1] * self.loss_bbox(
            pred_bboxes['offset'].corners[reg_inds, ...],
            gt_cors[reg_inds, ...], avg_factor=avg_factor)

        # loss_h2d = self.group_reg_loss_weights[4] * self.loss_bbox(
        #     pred_bboxes['h2d'][reg_inds, ...], gt_h2d[reg_inds, ...], avg_factor=avg_factor)
        if isinstance(self.loss_cls, GaussianVariFocalLoss):
            IOU3D = (pred_bboxes['iou'] + 0.3).clamp_max(1)
            loss_cls = self.loss_cls(
                (center2d_heatmap, pred_cls), (center_heatmap_target, IOU3D), avg_factor=avg_factor)
        else:
            loss_cls = self.loss_cls(center2d_heatmap, center_heatmap_target, avg_factor=avg_factor)
        loss_dict = dict(loss_cls=loss_cls, loss_bbox_dims=loss_bbox_dims, loss_bbox_depths=loss_bbox_depths,
                         iou3d=pred_bboxes['iou'].sum() / avg_factor,  # loss_h2d=loss_h2d,
                         loss_bbox_offsets=loss_bbox_offsets, loss_bbox_oris=loss_bbox_oris)

        return loss_dict

    def forward_train(self,
                      x,
                      img_metas,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_3d (list[Tensor]): 3D ground truth bboxes of the image,
                shape (num_gts, self.bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D ground truth labels of each box,
                shape (num_gts,).
            centers2d (list[Tensor]): Projected 3D center of each box,
                shape (num_gts, 2).
            depths (list[Tensor]): Depth of projected 3D center of each box,
                shape (num_gts,).
            attr_labels (list[Tensor]): Attribute labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)

        losses = self.loss(*outs, img_metas, **kwargs)

        return losses

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
