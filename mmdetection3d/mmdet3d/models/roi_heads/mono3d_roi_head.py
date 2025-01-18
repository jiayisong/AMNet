import torch
from abc import ABCMeta
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from mmcv.runner import BaseModule
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin


@HEADS.register_module()
class Mono3DRoIHead(BaseModule, metaclass=ABCMeta):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Mono3DRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
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

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        num_imgs = len(img_metas)
        rois_list = []
        iou_list = []
        index_list = []
        score_list = []
        label_list = []
        for img_id, bboxes in enumerate(proposal_list):
            bboxes, label, iou, index = bboxes
            if bboxes.size(0) > 0:
                img_inds = bboxes.new_full((bboxes.size(0),), img_id)
                rois = torch.cat([img_inds.unsqueeze(1), bboxes[:, :4]], dim=-1)
                score = bboxes[:, 4]
                index = img_inds * self.train_cfg.max_num_pre_img + index
            else:
                rois = bboxes.new_zeros((0, 5))
                score = bboxes.new_zeros((0,))
            rois_list.append(rois)
            iou_list.append(iou)
            index_list.append(index)
            score_list.append(score)
            label_list.append(label)
        rois = torch.cat(rois_list, 0)
        if rois.shape[0] == 0:
            return {'loss': 0}
        iou = torch.cat(iou_list, 0)
        index = torch.cat(index_list, 0)
        score = torch.cat(score_list, 0)
        label = torch.cat(label_list, 0)
        kwargs['iou'] = iou
        kwargs['index'] = index
        kwargs['score'] = score
        kwargs['label'] = label
        kwargs['rois'] = rois
        bbox_results = self._bbox_forward(x, rois)
        loss_bbox = self.bbox_head.loss(*bbox_results, **kwargs)
        loss_bbox['roi_num_per_img'] = torch.tensor(rois.shape[0] / num_imgs, dtype=torch.float32)
        return loss_bbox

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        '''
        center = 0.5 * (rois[:, 1:3] + rois[:, 3:5])
        wh = 20
        rois[:, 1:3] = center - wh
        rois[:, 3:5] = center + wh
        
        output = []
        rois = rois.long()
        y = torch.nn.functional.pad(x[0], (3, 3, 3, 3))
        for i in range(rois.shape[0]):
            roi = rois[i]
            im_idx = roi[0]
            bbox = roi[1:] / 8
            bbox = bbox.long()
            # 首先在feature上裁剪出roi的区域,naroow 类似于切片 下面详讲
            # 然后通过adaptive_max_pool2d 输出指定size的feature
            im = y.narrow(0, im_idx, 1)[..., bbox[1] + 3:bbox[3] + 3, bbox[0] + 3:bbox[2] + 3]
            output.append(im)
        bbox_feats = torch.cat(output, 0)
'''
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        # debug_bbox_feats = bbox_feats.detach().cpu().numpy()
        # debug_x = x[:self.bbox_roi_extractor.num_inputs][0].detach().cpu().numpy()
        # debug_rois = rois.detach().cpu().numpy()
        # debug_rois2 = debug_rois / 8
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

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False, **kwargs):
        """Test without augmentation."""
        '''
        num_imgs = len(img_metas)
        rois_list = []
        index_list = []
        score_list = []
        label_list = []
        num_proposals_per_img = []
        for img_id, bboxes in enumerate(proposal_list):
            bboxes, label, index = bboxes
            if bboxes.size(0) > 0:
                img_inds = bboxes.new_full((bboxes.size(0),), img_id)
                rois = torch.cat([img_inds.unsqueeze(1), bboxes[:, :4]], dim=-1)
                score = bboxes[:, 4]
                #index = img_inds * self.train_cfg.max_num_pre_img + index
            else:
                rois = bboxes.new_zeros((0, 5))
                score = bboxes.new_zeros((0,))
            rois_list.append(rois)
            index_list.append(index)
            score_list.append(score)
            label_list.append(label)
            num_proposals_per_img.append(len(bboxes))
        rois = torch.cat(rois_list, 0)
        index = torch.cat(index_list, 0)

        '''
        rois_list = []
        score_list = []
        label_list = []
        num_proposals_per_img = []
        for img_id, bboxes in enumerate(proposal_list):
            bboxes, label = bboxes
            if bboxes.size(0) > 0:
                img_inds = bboxes.new_full((bboxes.size(0),), img_id)
                rois = torch.cat([img_inds.unsqueeze(1), bboxes[:, :4]], dim=-1)
                score = bboxes[:, 4]
            else:
                rois = bboxes.new_zeros((0, 5))
                score = bboxes.new_zeros((0,))
            rois_list.append(rois)
            score_list.append(score)
            label_list.append(label)
            num_proposals_per_img.append(len(bboxes))
        rois = torch.cat(rois_list, 0)
        score = torch.cat(score_list, 0)

        if rois.shape[0] == 0:
            batch_size = len(proposal_list)
            bboxes = rois.new_zeros(0, 7)
            bboxes2d = rois.new_zeros(0, 5)
            scores = rois.new_zeros(0, )
            labels = rois.new_zeros((0,), dtype=torch.long)
            # There is no proposal in the whole batch
            return [(bboxes, scores, labels, None, bboxes2d, labels) for _ in range(batch_size)]
        bbox_results = self._bbox_forward(x, rois)
        bbox_results = [i.split(num_proposals_per_img, 0) for i in bbox_results]
        # apply bbox post-processing to each image individually
        result_list = []
        for i in range(len(proposal_list)):
            if rois_list[i].shape[0] == 0:
                # There is no proposal in the single image
                bboxes = rois.new_zeros(0, 7)
                bboxes2d = rois.new_zeros(0, 5)
                scores = rois.new_zeros(0, )
                labels = rois.new_zeros((0,), dtype=torch.long)
                labels2d = rois.new_zeros((0,), dtype=torch.long)
                attrs = None
            else:
                kwarg = {}
                bbox_result = [bbox[i] for bbox in bbox_results]
                for k, v in kwargs.items():
                    if not isinstance(v[0], list):
                        kwarg[k] = v[0][i].unsqueeze(0)
                bboxes, scores, labels, attrs, bboxes2d, labels2d = self.bbox_head.get_bboxes(*bbox_result,
                                                                                              img_metas[i],
                                                                                              rois_list[i],
                                                                                              score_list[i],
                                                                                              label_list[i],
                                                                                              # index_list[i],
                                                                                              rescale=rescale,
                                                                                              **kwarg)
            result_list.append((bboxes, scores, labels, attrs, bboxes2d, labels2d))
        return result_list

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
