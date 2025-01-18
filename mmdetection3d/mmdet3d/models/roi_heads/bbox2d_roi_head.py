import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from mmcv.runner import BaseModule
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, bbox_overlaps
from mmcv.cnn import bias_init_with_prob, xavier_init, build_conv_layer, build_norm_layer, ConvModule
from torch.nn import init


@HEADS.register_module()
class BBox2DRoIHead(BaseModule, metaclass=ABCMeta):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 aw_loss=False,
                 use_cls=None,
                 init_cfg=None):
        super(BBox2DRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.aw_loss = aw_loss
        if use_cls == 'cat':
            #self.weight = nn.Parameter(torch.ones([1, bbox_head['num_classes'], 1, 1]))
            self.weight = nn.Parameter(1/3 * torch.ones([1, bbox_head['num_classes'], bbox_roi_extractor['roi_layer']['output_size'], bbox_roi_extractor['roi_layer']['output_size']]))
            bbox_head['in_channels'] += bbox_head['num_classes']
        elif use_cls == 'add':
            self.weight = nn.Parameter(torch.zeros([bbox_head['num_classes'], bbox_head['in_channels']]))
            # self.weight = nn.Parameter(torch.zeros(
            #     [bbox_head['num_classes'], bbox_head['in_channels'], bbox_roi_extractor['roi_layer']['output_size'],
            #      bbox_roi_extractor['roi_layer']['output_size']]))
        elif use_cls == 'mul':
            self.weight = nn.Parameter(torch.ones([bbox_head['num_classes'], bbox_head['in_channels']]))
        elif use_cls == 'conv':
            self.weight = nn.Parameter(
                torch.ones([bbox_head['num_classes'], bbox_head['in_channels'], bbox_head['in_channels']]))
            self.bnrelu = nn.Sequential(
                build_norm_layer(bbox_head['norm_cfg'], bbox_head['in_channels'])[1],
                nn.ReLU(True),
            )
            for i in range(bbox_head['num_classes']):
                init.orthogonal_(self.weight[i], gain=1 / math.sqrt(3))
        elif use_cls == 'conv_last' or use_cls == 'conv_head' or use_cls == 'ach':
            bbox_head['use_cls'] = use_cls
        elif use_cls is None:
            pass
        else:
            raise RuntimeError('use_cls error')
        self.use_cls = use_cls
        self.init_bbox_head(bbox_roi_extractor, bbox_head)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'],)
        return outs

    # @profile
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      **kwargs):
        # assign gts and sample proposals
        num_imgs = len(img_metas)
        topk_bbox, topk_scores, topk_clses, index = proposal_list
        img_inds = topk_bbox[:, 0]
        index = img_inds * self.train_cfg.max_num_pre_img + index
        dwdh, dxdy, iou2d = self._bbox_forward(x, topk_bbox, topk_clses)
        rois1 = topk_bbox
        with torch.no_grad():
            index = index.long()
            # batch_index = batch_index.long()
            b, n, c = kwargs['cls_heatmap_pos'].shape
            cls_heatmap_pos = kwargs['cls_heatmap_pos'].view(b * n, c)[index, :]
            s1, label_gt = torch.max(cls_heatmap_pos, dim=1, keepdim=True)
            # s1_debug = s1.cpu().numpy()
            s1 = s1 * (label_gt == topk_clses.unsqueeze(1))
            # s1_debug2 = s1.cpu().numpy()
            # topk_scores_debug = topk_scores.detach().cpu().numpy()
            # topk_clses_debug = topk_clses.unsqueeze(1).detach().cpu().numpy()
            # label_gt_debug = label_gt.detach().cpu().numpy()
            # index_debug = index.detach().cpu().numpy()
            bbox2d_heatmap = kwargs['bbox2d_heatmap'].view(b * n, 4)[index, :]
            bbox2d = self.bbox_head.bbox_coder.decode_bbox2d(topk_bbox, dxdy, dwdh)
            iou_heatmap = bbox_overlaps(bbox2d.unsqueeze(0), bbox2d_heatmap.unsqueeze(0), mode='iou',
                                        is_aligned=True).squeeze(0).unsqueeze(1)
            gt = (topk_scores < 1 - 1e-3)
            bbox2d = torch.where(gt.unsqueeze(1), bbox2d, bbox2d_heatmap)
            # s2 = torch.where(gt.unsqueeze(1), iou_heatmap, iou_heatmap.new_ones((1,)))
            s2 = torch.where(gt.unsqueeze(1), iou2d, iou_heatmap.new_ones((1,)))
            w = bbox2d[:, 2] - bbox2d[:, 0]
            h = bbox2d[:, 3] - bbox2d[:, 1]
            valid_mask = (w >= self.bbox_head.train_cfg.min_bbox_size) & (h >= self.bbox_head.train_cfg.min_bbox_size)
            s2 = s2 * valid_mask.unsqueeze(1)
            rois2 = torch.cat((img_inds.unsqueeze(1), bbox2d), dim=1)
        if self.aw_loss:
            topk_scores = topk_scores.unsqueeze(1) * s1
            s1 = topk_scores.detach()
        loss_bbox = {}
        # loss_bbox['s1'] = torch.tensor(s1 / num_imgs, dtype=torch.float32)
        # loss_bbox['s2'] = torch.tensor(s2 / num_imgs,  dtype=torch.float32)
        proposal_list = rois1, rois2, topk_clses, s1, s2, index, topk_scores, dxdy, dwdh, iou2d, bbox2d_heatmap, iou_heatmap
        return loss_bbox, proposal_list

    # @profile
    def _bbox_forward(self, x, rois, cls=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        # bbox_feats1 = x[0][:, :, 0:5, 0:5]
        # bbox_feats2= x[0][:, :, 6:11, 0:5]
        # bbox_feats3 = x[0][:, :, 0:5, 9:14]
        # bbox_feats4 = x[0][:, :, 10:15, 10:15]
        # bbox_feats = torch.cat((bbox_feats1, bbox_feats2, bbox_feats3, bbox_feats4), 0)
        if cls is not None and self.use_cls is not None:
            if self.use_cls == 'cat':
                cls = F.one_hot(cls, self.bbox_head.num_classes).unsqueeze(2).unsqueeze(2) * self.weight
                cls = cls.expand(
                    [bbox_feats.shape[0], self.bbox_head.num_classes, bbox_feats.shape[2], bbox_feats.shape[3]])
                bbox_feats = torch.cat([bbox_feats, cls], dim=1)
                pred = self.bbox_head(bbox_feats)
            elif self.use_cls == 'add':
                cls = torch.index_select(self.weight, 0, index=cls)
                cls = cls.unsqueeze(2).unsqueeze(2)
                bbox_feats = bbox_feats + cls
                pred = self.bbox_head(bbox_feats)
            elif self.use_cls == 'mul':
                weight = self.weight
                # weight = torch.softmax(weight, dim=0)
                cls = torch.index_select(weight, 0, index=cls).unsqueeze(2).unsqueeze(2)
                cls = torch.sigmoid(cls)
                bbox_feats = bbox_feats * cls
                pred = self.bbox_head(bbox_feats)
            elif self.use_cls == 'conv':
                #B, C, H, W = bbox_feats.shape
                cls = torch.index_select(self.weight, 0, index=cls)
                bbox_feats = torch.einsum('bihw, bio->bohw', bbox_feats, cls)
                bbox_feats = self.bnrelu(bbox_feats)
                pred = self.bbox_head(bbox_feats)
            elif self.use_cls == 'conv_last' or self.use_cls == 'conv_head' or self.use_cls == 'ach':
                pred = self.bbox_head(bbox_feats, cls)
        else:
            pred = self.bbox_head(bbox_feats)
        return pred

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    # @profile
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False, **kwargs):
        topk_bbox, topk_scores, topk_clses = proposal_list
        kwargs['score'] = topk_scores
        kwargs['label'] = topk_clses
        kwargs['rois'] = topk_bbox
        bbox_results = self._bbox_forward(x, topk_bbox, topk_clses)
        rois_label_score_iou2d = self.bbox_head.get_bboxes(*bbox_results, img_metas, **kwargs)
        return rois_label_score_iou2d

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals
        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
