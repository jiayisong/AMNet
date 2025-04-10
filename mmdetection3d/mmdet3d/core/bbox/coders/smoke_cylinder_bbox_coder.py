import math

import numpy as np
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class SMOKECylinderCoder(BaseBBoxCoder):
    """Bbox Coder for SMOKE.

    Args:
        base_depth (tuple[float]): Depth references for decode box depth.
        base_dims (tuple[tuple[float]]): Dimension references [l, h, w]
            for decode box dimension for each category.
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, base_depth, base_offset, base_dims, code_size, down_ratio):
        super(SMOKECylinderCoder, self).__init__()
        self.base_depth = base_depth
        self.base_offset = base_offset
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.down_ratio = down_ratio

    def encode(self, locations, dimensions, orientations, input_metas, origin):
        """Encode CameraInstance3DBoxes by locations, dimensions, orientations.

        Args:
            locations (Tensor): Center location for 3D boxes.
                (N, 3)
            dimensions (Tensor): Dimensions for 3D boxes.
                shape (N, 3)
            orientations (Tensor): Orientations for 3D boxes.
                shape (N, 1)
            input_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Return:
            :obj:`CameraInstance3DBoxes`: 3D bboxes of batch images,
                shape (N, bbox_code_size).
        """

        bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        assert bboxes.shape[1] == self.bbox_code_size, 'bboxes shape dose not' \
                                                       'match the bbox_code_size.'
        batch_bboxes = input_metas[0]['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=origin)

        return batch_bboxes

    def decode(self,
               reg,
               points,
               labels,
               img2cam, K_out,
               gt_locations=None, gt_center2d=None, gt_depths=None):
        """Decode regression into locations, dimensions, orientations.

        Args:
            reg (Tensor): Batch regression for each predict center2d point.
                shape: (batch * K (max_objs), C)
            points(Tensor): Batch projected bbox centers on image plane.
                shape: (batch * K (max_objs) , 2)
            labels (Tensor): Batch predict class label for each predict
                center2d point.
                shape: (batch, K (max_objs))
            cam2imgs (Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)

            locations (None | Tensor): if locations is None, this function
                is used to decode while inference, otherwise, it's used while
                training using the ground truth 3d bbox locations.
                shape: (batch * K (max_objs), 3)

        Return:
            tuple(Tensor): The tuple has components below:
                - locations (Tensor): Centers of 3D boxes.
                    shape: (batch * K (max_objs), 3)
                - dimensions (Tensor): Dimensions of 3D boxes.
                    shape: (batch * K (max_objs), 3)
                - orientations (Tensor): Orientations of 3D
                    boxes.
                    shape: (batch * K (max_objs), 1)
        """
        depth_offsets = reg[:, 0]
        centers2d_offsets = reg[:, 1:3]
        dimensions_offsets = reg[:, 3:6]
        orientations = reg[:, 6:8]
        if gt_center2d is None:
            depths = self._decode_depth(depth_offsets)
            centers2d = self._decode_offset(centers2d_offsets, points)
            # get the 3D Bounding box's center location.
            pred_locations = self._decode_location(centers2d, depths, img2cam)
        else:
            depths = self._decode_depth(depth_offsets)
            centers2d = self._decode_offset(centers2d_offsets, points)
            pred_depth_locations = self._decode_location(gt_center2d, depths, img2cam)
            pred_centers2d_locations = self._decode_location(centers2d, gt_depths, img2cam)
            pred_locations = self._decode_location(centers2d, depths, img2cam)
            pred_locations = pred_depth_locations, pred_centers2d_locations, pred_locations
        pred_dimensions = self._decode_dimension(labels, dimensions_offsets)
        if gt_locations is None:
            pred_orientations = self._decode_orientation(
                orientations, pred_locations, K_out)
        else:
            pred_orientations = self._decode_orientation(
                orientations, gt_locations, K_out)

        return pred_locations, pred_dimensions, pred_orientations

    def _decode_depth(self, depth_offsets):
        """Transform depth offset to depth."""
        depths = depth_offsets * self.base_depth[1] + self.base_depth[0]
        return depths

    def _decode_offset(self, centers2d_offsets, points, scale=True):
        """Transform depth offset to depth."""
        centers2d_offsets = centers2d_offsets * self.base_offset[1] + self.base_offset[0]

        centers2d = (points + centers2d_offsets)
        if scale:
            centers2d = centers2d * self.down_ratio
        return centers2d

    def _decode_location(self, centers2d, depths, img2cams):
        """Retrieve objects location in camera coordinate based on projected
        points.

        Args:
            points (Tensor): Projected points on feature map in (x, y)
                shape: (batch * K, 2)
            centers2d_offset (Tensor): Project points offset in
                (delta_x, delta_y). shape: (batch * K, 2)
            depths (Tensor): Object depth z.
                shape: (batch * K)
            cam2imgs (Tensor): Batch camera intrinsics matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            trans_mats (Tensor): transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
        """
        # number of points
        N = centers2d.shape[0]
        # batch_size
        N_batch = img2cams.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()

        img2cams = img2cams[obj_id]

        centers2d_extend = torch.cat((centers2d, centers2d.new_ones(N, 1)), dim=1)
        # expand project points as [N, 3, 1]
        centers2d_img = centers2d_extend.unsqueeze(-1)

        centers2d_img = centers2d_img

        cam2imgs_inv = img2cams[:, :3, :3]

        locations = torch.matmul(cam2imgs_inv, centers2d_img).squeeze(2)
        theta = locations[:, :1]
        yy = locations[:, 1:2]
        xyz = torch.cat((torch.sin(theta), yy, torch.cos(theta)), dim=1) * depths.view(N, 1)
        if img2cams.shape[1] == 4:
            xyz = xyz + img2cams[:, :3, 3]
        return xyz

    def _decode_dimension(self, labels, dims_offset):
        """Transform dimension offsets to dimension according to its category.

        Args:
            labels (Tensor): Each points' category id.
                shape: (N, K)
            dims_offset (Tensor): Dimension offsets.
                shape: (N, 3)
        """
        labels = labels.flatten().long()
        base_dims = dims_offset.new_tensor(self.base_dims)
        dims_select = base_dims[:, labels, :]
        dimensions = (dims_offset * dims_select[1]).exp() * dims_select[0]
        return dimensions

    def _decode_orientation(self, ori_vector, locations, K_out):
        """Retrieve object orientation.

        Args:
            ori_vector (Tensor): Local orientation in [sin, cos] format.
                shape: (N, 2)
            locations (Tensor): Object location.
                shape: (N, 3)

        Return:
            Tensor: yaw(Orientation). Notice that the yaw's
                range is [-np.pi, np.pi].
                shape：(N, 1）
        """
        assert len(ori_vector) == len(locations)

        # number of points
        N = ori_vector.shape[0]
        # batch_size
        N_batch = K_out.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()

        K_out = K_out[obj_id]

        locations = locations.view(-1, 3) + K_out

        rays = torch.atan2(locations[:, 0], locations[:, 2])
        alphas = torch.atan2(ori_vector[:, 0], ori_vector[:, 1])

        # get cosine value positive and negative index.
        # cos_pos_inds = (ori_vector[:, 1] >= 0).nonzero(as_tuple=False)
        # cos_neg_inds = (ori_vector[:, 1] < 0).nonzero(as_tuple=False)

        # alphas[cos_pos_inds] -= np.pi / 2
        # alphas[cos_neg_inds] += np.pi / 2
        # retrieve object rotation y angle.
        yaws = (alphas + rays + math.pi) % (2 * math.pi) - math.pi

        # larger_inds = (yaws > np.pi).nonzero(as_tuple=False)
        # small_inds = (yaws < -np.pi).nonzero(as_tuple=False)

        # if len(larger_inds) != 0:
        #    yaws[larger_inds] -= 2 * np.pi
        # if len(small_inds) != 0:
        #    yaws[small_inds] += 2 * np.pi

        yaws = yaws.unsqueeze(-1)
        return yaws
