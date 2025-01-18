# Copyright (c) OpenMMLab. All rights reserved.
import time

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
from collections import OrderedDict
import torch.distributed as dist
import warnings


@DETECTORS.register_module()
class TwoStageMono3DDetector(TwoStageDetector):
    """Base class for monocular 3D single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_head_ = rpn_head.copy()
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # @profile
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # @profile
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        # print(losses)
        # print('s')
        # print(losses)
        log_vars = OrderedDict()

        loss = losses.pop('loss')
        # '''
        # print('s', loss.is_cuda, loss.data, loss.layout, loss.shape)
        log_vars['loss'] = loss.item()
        # print('m')
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.item()

        # print('f')
        # '''
        return loss, log_vars

    # @profile
    def forward_train(self,
                      img,
                      img_metas,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        rois1, rois2, label, s1, s2, index, topk_scores, dxdy, dwdh, iou2d, bbox2d_heatmap, iou_heatmap = proposal_list
        roi_losses, s3 = self.roi_head.forward_train(x, img_metas, (rois2, label, s1, s2, index), **kwargs)
        # loss_rpn = self.rpn_head.loss(topk_scores, s2 * s3 * 0, s2 * 0)
        loss_bbox2d = self.rpn_head.roi_head.bbox_head.loss(dwdh, dxdy, iou2d, rois1, bbox2d_heatmap, iou_heatmap,
                                                            s1)
        # s1_debug = s1.cpu().numpy()
        # s2_debug = s2.cpu().numpy()
        # s3_debug = s3.cpu().numpy()
        loss = roi_losses.pop('loss')
        losses.update(roi_losses)
        losses['loss'] = losses['loss'] + loss
        loss = loss_bbox2d.pop('loss')
        losses.update(loss_bbox2d)
        losses['loss'] = losses['loss'] + loss
        # loss = loss_rpn.pop('loss')
        # losses['loss'] = losses['loss'] + loss
        # losses['s1'] = s1.sum()
        # losses['s2'] = s2.sum()
        # losses['s3'] = s3.sum()

        # sclass = s1[(s1 < 0.999) * (s1 > 0)]
        # s2d = (s1 * s2)[((s1 * s2) < 0.999) * ((s1 * s2) > 0)]
        #
        # losses['sclass_q0'] = sclass.min()
        # losses['sclass_q1'] = torch.quantile(sclass, 0.25)
        # losses['sclass_q2'] = torch.quantile(sclass, 0.5)
        # losses['sclass_q3'] = torch.quantile(sclass, 0.75)
        # losses['sclass_q4'] = sclass.max()
        #
        # losses['s2d_q0'] = s2d.min()
        # losses['s2d_q1'] = torch.quantile(s2d, 0.25)
        # losses['s2d_q2'] = torch.quantile(s2d, 0.5)
        # losses['s2d_q3'] = torch.quantile(s2d, 0.75)
        # losses['s2d_q4'] = s2d.max()

        return losses

    # @profile
    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas, **kwargs)
        else:
            proposal_list = proposals
        bbox_outputs = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        from mmdet.core import bbox2result
        for i, (bboxes, scores, labels, attrs, bboxes2d, labels2d) in enumerate(bbox_outputs):
            bbox_list[i]['img_bbox2d'] = bbox2result(bboxes2d, labels2d, self.roi_head.bbox_head.num_classes)
            bbox_list[i]['img_bbox'] = bbox3d2result(bboxes, scores, labels, attrs)
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """

        imgs = torch.cat(imgs, 0)
        img_metas = img_metas[0] + img_metas[1]
        kwargs = {k: torch.cat(v, 0) for k, v in kwargs.items()}
        x = self.extract_feat(imgs)
        #x1, x2 = torch.chunk(x[0], 2, dim=0)
        #x1, x2 = 0.5 * (torch.flip(x2, [3]) + x1), 0.5 * (torch.flip(x1, [3]) + x2)
        #x = [torch.cat([x1, x2], 0),]
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas, **kwargs)
        bbox_outputs = self.roi_head.aug_test(x, proposal_list, img_metas, rescale=rescale, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas) // 2)]
        from mmdet.core import bbox2result
        for i, (bboxes, scores, labels, attrs, bboxes2d, labels2d) in enumerate(bbox_outputs):
            bbox_list[i]['img_bbox2d'] = bbox2result(bboxes2d, labels2d, self.roi_head.bbox_head.num_classes)
            bbox_list[i]['img_bbox'] = bbox3d2result(bboxes, scores, labels, attrs)
        return bbox_list

    def show_results(self, data, result, out_dir, show=False, score_thr=None):
        """Results visualization.

        Args:
            data (list[dict]): Input images and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
            show (bool, optional): Determines whether you are
                going to show result by open3d.
                Defaults to False.
            TODO: implement score_thr of single_stage_mono3d.
            score_thr (float, optional): Score threshold of bounding boxes.
                Default to None.
                Not implemented yet, but it is here for unification.
        """
        for batch_id in range(len(result)):
            if isinstance(data['img_metas'][0], DC):
                img_filename = data['img_metas'][0]._data[0][batch_id][
                    'filename']
                cam2img = data['img_metas'][0]._data[0][batch_id]['cam2img_ori']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                img_filename = data['img_metas'][0][batch_id]['filename']
                cam2img = data['img_metas'][0][batch_id]['cam2img_ori']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            img = mmcv.imread(img_filename)
            file_name = osp.split(img_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[batch_id]['img_bbox']['boxes_3d']
            if 'gt_bboxes_3d' in data:
                gt_bboxes = data['gt_bboxes_3d'][0]._data[0][batch_id]
                assert isinstance(gt_bboxes, CameraInstance3DBoxes), \
                    f'unsupported predicted bbox type {type(gt_bboxes)}'
            else:
                gt_bboxes = None
            assert isinstance(pred_bboxes, CameraInstance3DBoxes), \
                f'unsupported predicted bbox type {type(pred_bboxes)}'

            if score_thr is not None:
                pred_scores = result[batch_id]['img_bbox']['scores_3d']
                pred_bboxes = pred_bboxes[pred_scores > score_thr]
            show_multi_modality_result(
                img,
                gt_bboxes,
                pred_bboxes,
                cam2img,
                out_dir,
                file_name,
                'camera',
                show=show)

    def forward_test(self, imgs, img_metas, rescale, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            for k in kwargs.keys():
                kwargs[k] = kwargs[k][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)
