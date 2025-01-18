import torch
from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmdet.models.builder import DETECTORS
from .single_stage_mono3d import SingleStageMono3DDetector
from collections import OrderedDict
import torch.distributed as dist


@DETECTORS.register_module()
class MySingleStageMono3DDetector(SingleStageMono3DDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MySingleStageMono3DDetector, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained,
                                                          init_cfg)

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
        log_vars = OrderedDict()
        loss = losses.pop('loss')
        log_vars['loss'] = loss
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
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
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            for i, (bboxes, scores, labels, attrs, bboxes2d, labels2d) in enumerate(bbox_outputs):
                bbox_list[i]['img_bbox2d'] = bbox2result(bboxes2d, labels2d, self.bbox_head.num_classes)
                bbox_list[i]['img_bbox'] = bbox3d2result(bboxes, scores, labels, attrs)
        else:
            for i, (bboxes, scores, labels, attrs) in enumerate(bbox_outputs):
                bbox_list[i]['img_bbox'] = bbox3d2result(bboxes, scores, labels, attrs)
        return bbox_list

    # @profile
    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, **kwargs)
        return losses
