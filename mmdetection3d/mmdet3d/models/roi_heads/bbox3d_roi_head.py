import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from mmcv.runner import BaseModule
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin
from .bbox2d_roi_head import BBox2DRoIHead


@HEADS.register_module()
class BBox3DRoIHead(BBox2DRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    # @profile
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      **kwargs):
        # assign gts and sample proposals
        rois, label, s1, s2, index = proposal_list
        bbox_results = self._bbox_forward(x, rois, label)
        if self.aw_loss:
            s12 = s1 * s2
        else:
            s12 = s1
        loss_bbox, s3 = self.bbox_head.loss(*bbox_results, rois, label, s12, index, **kwargs)
        return loss_bbox, s3

    # @profile
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False, **kwargs):
        rois, label, score, iou2d = proposal_list
        bbox_results = self._bbox_forward(x, rois, label)
        num_imgs = len(img_metas)
        bbox_resultss = [torch.chunk(j, num_imgs, dim=0) if j is not None else None for j in bbox_results]
        rois_list = torch.chunk(rois, num_imgs, dim=0)
        score_list = torch.chunk(score, num_imgs, dim=0)
        label_list = torch.chunk(label, num_imgs, dim=0)
        '''
        img_id = rois[:, 0]
        bbox_resultss = []
        rois_list = []
        score_list = []
        label_list = []
        for i in range(num_imgs):
            a = (img_id == i)
            bbox_resultss.append([j[a,:] if j is not None else None for j in bbox_results])
            rois_list.append(rois[a,:])
            score_list.append(score[a])
            label_list.append(label[a])
        '''
        result_list = []
        for i in range(num_imgs):
            if rois_list[i].shape[0] == 0:
                # There is no proposal in the single image
                bboxes = rois.new_zeros(0, 7)
                bboxes2d = rois.new_zeros(0, 5)
                scores = rois.new_zeros(0, )
                labels = rois.new_zeros((0,), dtype=torch.long)
                labels2d = rois.new_zeros((0,), dtype=torch.long)
            else:
                kwarg = {}
                bbox_result = [res[i] if res is not None else None for res in bbox_resultss]
                for k, v in kwargs.items():
                    if isinstance(v[i], torch.Tensor):
                        kwarg[k] = v[i].unsqueeze(0)
                bboxes, scores, labels, bboxes2d, _, labels2d = self.bbox_head.get_bboxes(*bbox_result,
                                                                                          img_metas[i],
                                                                                          rois_list[i],
                                                                                          score_list[i],
                                                                                          label_list[i],
                                                                                          # index_list[i],
                                                                                          rescale=rescale,
                                                                                          with_nms=True,
                                                                                          **kwarg)
                bboxes = img_metas[i]['box_type_3d'](bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
            result_list.append((bboxes, scores, labels, None, bboxes2d, labels2d))
        return result_list

    def aug_test(self,
                 x,
                 proposal_list,
                 img_metas,
                 proposals=None,
                 rescale=False, **kwargs):
        rois, label, score, iou2d = proposal_list
        bbox_results = self._bbox_forward(x, rois, label)
        num_imgs = len(img_metas)
        bbox_resultss = [torch.chunk(j, num_imgs, dim=0) if j is not None else None for j in bbox_results]
        rois_list = torch.chunk(rois, num_imgs, dim=0)
        score_list = torch.chunk(score, num_imgs, dim=0)
        label_list = torch.chunk(label, num_imgs, dim=0)
        result_list = []
        for i in range(num_imgs):
            if rois_list[i].shape[0] == 0:
                # There is no proposal in the single image
                bboxes = rois.new_zeros(0, 7)
                bboxes2d = rois.new_zeros(0, 5)
                scores = rois.new_zeros(0, )
                scores2d = rois.new_zeros(0, )
                labels = rois.new_zeros((0,), dtype=torch.long)
                labels2d = rois.new_zeros((0,), dtype=torch.long)
            else:
                kwarg = {}
                bbox_result = [res[i] if res is not None else None for res in bbox_resultss]
                for k, v in kwargs.items():
                    kwarg[k] = v[i].unsqueeze(0)
                bboxes, scores, labels, bboxes2d, scores2d, labels2d = self.bbox_head.get_bboxes(*bbox_result,
                                                                                                 img_metas[i],
                                                                                                 rois_list[i],
                                                                                                 score_list[i],
                                                                                                 label_list[i],
                                                                                                 # index_list[i],
                                                                                                 rescale=rescale,
                                                                                                 with_nms=False,
                                                                                                 **kwarg)
            result_list.append((bboxes, scores, labels, bboxes2d, scores2d, labels2d))
        fuse_result = []
        for i in range(num_imgs // 2):
            bboxes_1, scores_1, labels_1, bboxes2d_1, scores2d_1, labels2d_1 = result_list[i]
            bboxes_2, scores_2, labels_2, bboxes2d_2, scores2d_2, labels2d_2 = result_list[i + num_imgs // 2]
            bboxes = torch.cat([bboxes_1, bboxes_2], 0)
            scores = torch.cat([scores_1, scores_2], 0)
            labels = torch.cat([labels_1, labels_2], 0)
            bboxes2d = torch.cat([bboxes2d_1, bboxes2d_2], 0)
            scores2d = torch.cat([scores2d_1, scores2d_2], 0)
            labels2d = torch.cat([labels2d_1, labels2d_2], 0)
            # bboxes = bboxes_1
            # scores = scores_1
            # labels = labels_1
            # bboxes2d = bboxes2d_1
            # scores2d = scores2d_1
            # labels2d = labels2d_1
            # bboxes = bboxes_2
            # scores = scores_2
            # labels = labels_2
            # bboxes2d = bboxes2d_2
            # scores2d = scores2d_2
            # labels2d = labels2d_2
            if bboxes2d.numel() > 0:
                bboxes2d, labels2d, keep1 = self.bbox_head.box2d_nms(bboxes2d, scores2d, labels2d)
            else:
                bboxes2d = bboxes2d.new_zeros(0, 5)
            if bboxes.numel() > 0:
                bboxes, scores, labels, keep2 = self.bbox_head.box3d_nms(bboxes, scores, labels)
            else:
                bboxes = bboxes.new_zeros(0, 7)
            bboxes = img_metas[i]['box_type_3d'](bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
            fuse_result.append((bboxes, scores, labels, None, bboxes2d, labels2d))
        return fuse_result

