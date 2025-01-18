import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair, _single
from mmcv.cnn.utils import constant_init, kaiming_init, normal_init, uniform_init
from mmcv.cnn import bias_init_with_prob, xavier_init, build_conv_layer, build_norm_layer, ConvModule
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from mmcv.ops import batched_nms, box_iou_rotated, ModulatedDeformConv2dPack
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet3d.core import xywhr2xyxyr
from torch.nn.parameter import Parameter
from torch.nn import init
from mmcv.cnn import CONV_LAYERS

HEAD_INIT_BOUND = 0.01


class ACH3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACH3, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels),
                                nn.ReLU(True),
                                nn.Linear(in_channels, out_channels),
                                # nn.ReLU(True),
                                )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        init.orthogonal_(self.fc[0].weight, gain=1 / math.sqrt(3))
        init.orthogonal_(self.fc[2].weight, gain=1 / math.sqrt(3))
        init.constant_(self.fc[0].bias, 0)
        init.constant_(self.fc[2].bias, 0)

    def forward(self, x, cls=None):
        y = x
        if cls is not None:
            y = torch.cat([y, cls], 1)
        else:
            y = y
        y = self.fc[0](y)
        y = self.fc[1](y)
        y = self.fc[2](y)
        #x = y
        x = F.relu(x + y, True)
        # x = x + self.fc(x)
        return x


@CONV_LAYERS.register_module('ACH1')
class ACH1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 **kwargs):
        super(ACH1, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.coupling_weight = nn.Parameter(torch.zeros([1, ]))
        self.register_buffer('ones', torch.ones([1, ]))
        self.out_channels_per_groups = out_channels // groups
        self.in_channels_per_groups = in_channels // groups
        self.weight = Parameter(torch.empty((groups, out_channels // groups, in_channels, *self.kernel_size), ))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, ))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        k = self.in_channels * self.kernel_size[0] * self.kernel_size[1] / self.groups
        init.uniform_(self.weight, -1 / math.sqrt(k), 1 / math.sqrt(k))
        if self.bias is not None:
            init.constant_(self.bias, 0)

    def forward(self, x):
        coupling_weight = torch.sigmoid(self.coupling_weight)
        temp = torch.cat([self.ones.expand(self.in_channels_per_groups, ),
                          coupling_weight.expand(self.in_channels_per_groups * self.groups, )], 0)
        temp = temp.view(1, - 1).expand(self.groups - 1, self.in_channels_per_groups * (1 + self.groups)).reshape(-1)
        temp = torch.cat([temp, self.ones.expand(self.in_channels_per_groups, )], 0).view(self.groups, 1,
                                                                                          self.in_channels, 1, 1)
        weight = (self.weight * temp).view(self.out_channels, self.in_channels, *self.kernel_size)
        y = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, 1)
        return y


@CONV_LAYERS.register_module('ACH2')
class ACH2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 **kwargs):
        super(ACH2, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        # self.coupling_weight = nn.Parameter(torch.zeros([self.groups, ]))
        self.coupling_weight = nn.Sequential(nn.Linear(in_channels, in_channels // 4),
                                             nn.ReLU(True),
                                             nn.Linear(in_channels // 4, 1),
                                             nn.Sigmoid())
        self.register_buffer('ones', torch.ones([1, 1]))
        self.out_channels_per_groups = out_channels // groups
        self.in_channels_per_groups = in_channels // groups
        self.weight = Parameter(torch.empty((1, groups, out_channels // groups, in_channels, *self.kernel_size), ))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, ))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        k = self.in_channels * self.kernel_size[0] * self.kernel_size[1] / self.groups
        init.uniform_(self.weight, -1 / math.sqrt(k), 1 / math.sqrt(k))
        if self.bias is not None:
            init.constant_(self.bias, 0)

    def forward(self, x):
        coupling_weight = self.coupling_weight(torch.mean(x, dim=[2, 3], keepdim=False))
        B, _ = coupling_weight.shape
        temp = torch.cat([self.ones.expand(B, self.in_channels_per_groups),
                          coupling_weight.expand(B, self.in_channels_per_groups * self.groups)], 1)
        temp = temp.view(B, 1, - 1).expand(B, self.groups - 1, self.in_channels_per_groups * (1 + self.groups)).reshape(
            B, -1)
        temp = torch.cat([temp, self.ones.expand(B, self.in_channels_per_groups, )], 1).view(B, self.groups, 1,
                                                                                             self.in_channels, 1, 1)
        weight = (self.weight * temp).view(self.out_channels, self.in_channels, *self.kernel_size)
        y = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, 1)
        return y


class TorchInitConvModule(ConvModule):
    # '''
    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            # uniform_init(self.conv, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
            # kaiming_init(self.conv, a=0, nonlinearity='relu', mode='fan_in', bias=0, distribution='uniform')
            # normal_init(self.conv, 0, HEAD_INIT_BOUND)
            init.orthogonal_(self.conv.weight, gain=1 / math.sqrt(3))
            # kaiming_init(self.conv, a=math.sqrt(5), nonlinearity='leaky_relu', mode='fan_in', bias=0, distribution='uniform')
            # pass
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)
    # '''


def init_head(m, a):
    for i in m.modules():
        # nn.init.uniform_(m[-1].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        if hasattr(i, 'weight'):
            init.orthogonal_(i.weight, gain=0.05)
        # init.constant_(m[-1].weight, 0)
        if hasattr(i, 'bias'):
            init.constant_(i.bias, a)


class SAM(BaseModule):
    def __init__(self, in_channel, out_channel=1, groups=1, kernel=3, init_cfg=None):
        super(SAM, self).__init__(init_cfg)
        assert out_channel % groups == 0 and in_channel % groups == 0
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel, 1, kernel // 2),
                                  nn.Sigmoid())
        self.out_channel = out_channel
        self.groups = groups
        self.init_weights()

    def init_weights(self):
        init.constant_(self.conv[0].weight, 0)
        init.constant_(self.conv[0].bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.conv(x).view(b, self.out_channel // self.groups, self.groups, 1, h, w)
        y = y * x.view(b, 1, self.groups, c // self.groups, h, w)
        y = y.view(b, -1, h, w)
        return y


class ConvPool(BaseModule):
    def __init__(self, in_channel, feat_channel, norm_cfg, roi_feat_size, group=1, init_cfg=None, conv_cfg=None):
        super(ConvPool, self).__init__(init_cfg)
        # conv_cfg = dict(type='ACH1')  if group > 1 else None
        conv_cfg = None
        self.convpool = nn.Sequential(  # CBAM(in_channel),
            # SAM(in_channel, group),
            # TorchInitConvModule(in_channel, feat_channel, 1, 1, 0, groups=1, norm_cfg=norm_cfg),
            # TorchInitConvModule(in_channel * group, feat_channel, 3, 1, 0, groups=group, norm_cfg=norm_cfg),
            TorchInitConvModule(in_channel, feat_channel, 3, 1, 0, groups=1, norm_cfg=norm_cfg),
            *[TorchInitConvModule(feat_channel, feat_channel, 3, 1, 0, groups=group, norm_cfg=norm_cfg,
                                  conv_cfg=conv_cfg) for _ in
              range(roi_feat_size // 2 - 1)],
            nn.Flatten(),
        )
        '''
        self.convpool2 = nn.Sequential(  # CBAM(in_channel),
            TorchInitConvModule(in_channel, feat_channel, (roi_feat_size, 1), 1, 0, norm_cfg=norm_cfg),
            # SAM(feat_channel),
            TorchInitConvModule(feat_channel, feat_channel, (1, roi_feat_size), 1, 0, norm_cfg=norm_cfg),
            # nn.AvgPool2d(roi_feat_size, 1),
            nn.Flatten()
        )
        self.convpool3 = nn.Sequential(  # CBAM(in_channel),
            TorchInitConvModule(in_channel, feat_channel, (1, roi_feat_size), 1, 0, norm_cfg=norm_cfg),
            # SAM(feat_channel),
            TorchInitConvModule(feat_channel, feat_channel, (roi_feat_size, 1), 1, 0, norm_cfg=norm_cfg),
            # nn.AvgPool2d(roi_feat_size, 1),
            nn.Flatten()
        )
        '''

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nonlinearity = 'leaky_relu'
                a = math.sqrt(5)
                mode = 'fan_out'
                kaiming_init(m, a=a, nonlinearity=nonlinearity, mode=mode, bias=0, distribution='uniform')
                # assert False

    def forward(self, x):
        y1 = self.convpool(x)
        y = y1
        # y2 = self.convpool2(x)
        # y3 = self.convpool3(x)
        # y = (y1 + y2 + y3) / 3
        # y = (y2 + y3) / 2
        return y


@HEADS.register_module()
class BBox2DHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 roi_feat_size, in_channels, feat_channel, num_classes, pred, bbox_coder, loss_wh, loss_offset,
                 loss_iou2d, loss_iou, norm_cfg, use_cls=None,
                 test_cfg=None, train_cfg=None, init_cfg=None):
        super(BBox2DHead, self).__init__(init_cfg)
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.in_channels = in_channels
        self.pred_iou2d = 'iou2d' in pred
        self.pred_iou = 'iou' in pred
        self.pred_offset = 'offset' in pred
        self.pred_wh = 'wh' in pred
        if 'union' in pred:
            self.pred_iou = True
            self.pred_offset = True
            self.pred_wh =True
            self.pred_union = True
        else:
            self.pred_union = False
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_iou = build_loss(loss_iou)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.num_classes = num_classes
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.fp16_enabled = False
        self.use_cls = use_cls
        if use_cls == 'conv_head':
            self.stem = nn.Sequential(
                # SAM(in_channels),
                ConvPool(in_channels, feat_channel * num_classes, norm_cfg, roi_feat_size, group=num_classes)
            )
            self.wh_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 2),
            ) for _ in range(num_classes)])
            self.offset_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 2),
            ) for _ in range(num_classes)])
            self.iou_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 1),
            ) for _ in range(num_classes)])
        elif use_cls == 'conv_last':
            self.stem = nn.Sequential(
                # SAM(in_channels),
                ConvPool(in_channels, feat_channel, norm_cfg, roi_feat_size, group=1)
            )
            self.wh_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 2),
            ) for _ in range(num_classes)])
            self.offset_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 2),
            ) for _ in range(num_classes)])
            self.iou_head = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_channel, 1),
            ) for _ in range(num_classes)])
        # feat_channel = in_channels
        else:
            self.stem = nn.Sequential(
                # SAM(in_channels),
                ConvPool(in_channels, feat_channel, norm_cfg, roi_feat_size, group=1)
            )
            self.wh_head = nn.Sequential(
                nn.Linear(feat_channel, 2),
            )
            self.offset_head = nn.Sequential(
                nn.Linear(feat_channel, 2),
            )
            self.iou_head = nn.Sequential(
                nn.Linear(feat_channel, 1),
            )

    # @auto_fp16()
    # @profile
    def forward(self, x, cls=None):
        x = self.stem(x)
        B, _ = x.shape
        if self.use_cls == 'conv_head':
            x = torch.chunk(x, self.num_classes, 1)
            wh_pred = []
            offset_pred = []
            iou_pred = []
            for i, wh, oh, ih in zip(x, self.wh_head, self.offset_head, self.iou_head):
                wh_pred.append(wh(i))
                offset_pred.append(oh(i))
                iou_pred.append(ih(i).sigmoid())
            wh_pred = torch.stack(wh_pred, 1)
            offset_pred = torch.stack(offset_pred, 1)
            iou_pred = torch.stack(iou_pred, 1)
            cls = F.one_hot(cls, self.num_classes).unsqueeze(1).float()
            wh_pred = torch.bmm(cls, wh_pred).view(B, -1)
            offset_pred = torch.bmm(cls, offset_pred).view(B, -1)
            iou_pred = torch.bmm(cls, iou_pred).view(B, -1)
        elif self.use_cls == 'conv_last':
            wh_pred = []
            offset_pred = []
            iou_pred = []
            for wh, oh, ih in zip(self.wh_head, self.offset_head, self.iou_head):
                wh_pred.append(wh(x))
                offset_pred.append(oh(x))
                iou_pred.append(ih(x).sigmoid())
            wh_pred = torch.stack(wh_pred, 1)
            offset_pred = torch.stack(offset_pred, 1)
            iou_pred = torch.stack(iou_pred, 1)
            cls = F.one_hot(cls, self.num_classes).unsqueeze(1).float()
            wh_pred = torch.bmm(cls, wh_pred).view(B, -1)
            offset_pred = torch.bmm(cls, offset_pred).view(B, -1)
            iou_pred = torch.bmm(cls, iou_pred).view(B, -1)
        else:
            wh_pred = self.wh_head(x)
            offset_pred = self.offset_head(x)
            iou_pred = self.iou_head(x).sigmoid()
        return wh_pred, offset_pred, iou_pred

    def init_weights(self):
        init_head(self.wh_head, 0)
        init_head(self.offset_head, 0)
        init_head(self.iou_head, -4.6)

    def get_targets(self, index, batch_index, **kwargs):
        index = index.long()
        # batch_index = batch_index.long()
        b, n, c = kwargs['cls_heatmap_pos'].shape
        cls_heatmap_pos = kwargs['cls_heatmap_pos'].view(b * n, c)[index, :]
        # cls_heatmap_neg = kwargs['cls_heatmap_neg'].view(b * n, c)[index, :]
        bbox2d_heatmap = kwargs['bbox2d_heatmap'].view(b * n, 4)[index, :]
        return bbox2d_heatmap, cls_heatmap_pos

    @force_fp32(apply_to=('wh_pred', 'offset_pred', 'iou_pred'))
    # @profile
    def loss(self, wh_pred, offset_pred, iou_pred,
             rois, bbox2d_heatmap, iou_heatmap, mask, **kwargs):
        # bbox2d_heatmap, cls_heatmap_pos = self.get_targets(index, rois[:, 0], **kwargs)
        # mask, label_gt = torch.max(cls_heatmap_pos, dim=1, keepdim=True)
        # mask = mask * score.unsqueeze(1)
        # mask = mask * (score.unsqueeze(1) > 1 - 1e-3)
        loss = 0
        outputs = dict()

        with torch.no_grad():
            dxdy, dwdh = self.bbox_coder.encode_bbox2d(rois, bbox2d_heatmap)
        # dwdh_debug = dwdh.detach().cpu().numpy()
        # print((dwdh * (mask > 0.991)).min(), (dwdh * (mask > 0.991)).max(), (dwdh * (mask > 0.991)).sum() / (mask > 0.991).sum())
        # print((dxdy * (mask > 0.991)).min(), (dxdy * (mask > 0.991)).max(), (dxdy * (mask > 0.991)).sum() / (mask > 0.991).sum())
        if self.pred_union:
            mask_l1 = mask * (1 - iou_heatmap**0.1)
            mask_iou = mask * (iou_heatmap**0.1)
        else:
            mask_l1 = mask
            mask_iou = mask
        if self.pred_wh:
            loss_2d_wh, loss_2d_wh_show = self.loss_wh(wh_pred, dwdh, mask_l1)
            loss = loss + loss_2d_wh
            for key, value in loss_2d_wh_show.items():
                outputs[f'wh_{key}'] = value
        if self.pred_offset:
            loss_2d_offset, loss_2d_offset_show = self.loss_offset(offset_pred, dxdy, mask_l1)
            loss = loss + loss_2d_offset
            for key, value in loss_2d_offset_show.items():
                outputs[f'offset_{key}'] = value
        if self.pred_iou:
            # offset_pred, wh_pred = dxdy, dwdh
            bbox2d = self.bbox_coder.decode_bbox2d(rois, offset_pred, wh_pred)
            loss_2d_iou, loss_2d_iou_show = self.loss_iou(bbox2d, bbox2d_heatmap, mask_iou)
            loss = loss + loss_2d_iou
            for key, value in loss_2d_iou_show.items():
                outputs[f'2diouloss_{key}'] = value

        '''
        if torch.any(torch.isinf(loss_2d_wh)):
            raise RuntimeError('2d wh loss is nan')
        if torch.any(torch.isinf(loss_2d_offset)):
            raise RuntimeError('2d offset loss is nan')
        '''

        outputs['iou2d'] = (iou_heatmap * mask).sum() / mask.sum()
        if self.pred_iou2d:
            loss_iou2d, loss_iou_show = self.loss_iou2d(iou_pred, iou_heatmap, mask)
            loss = loss + loss_iou2d
            for key, value in loss_iou_show.items():
                outputs[f'iou_{key}'] = value
            # iou_heatmap = iou_pred.squeeze(1)
        outputs['loss'] = loss
        return outputs

    @force_fp32(apply_to=('wh_pred', 'offset_pred', 'iou_pred'))
    # @profile
    def get_bboxes(self, wh_pred, offset_pred, iou_pred,
                   img_metas, rois, score, label, collect=(), rescale=False, **kwargs):
        # xy_max = kwargs['xy_max']
        # xy_min = kwargs['xy_min']
        # b, _ = rois.shape
        # score = score.detach()
        if len(collect) > 0:
            cfg = self.train_cfg
            add_gt = cfg.add_gt
        else:
            cfg = self.test_cfg
            add_gt = False
        merge = [kwargs[i] for i in collect]
        bbox2d = self.bbox_coder.decode_bbox2d(rois, offset_pred, wh_pred)
        img_inds = rois[:, 0]
        if add_gt:
            gt = (score < 1 - 1e-3)
            '''
            #temp = (torch.arange(0, gt.size(0), device=gt.device, dtype=torch.long))
            #gt = temp[gt]
            bbox2d_gt = kwargs['bbox2d_gt'][gt, :]
            img_inds_gt = img_inds[gt]
            roi_gt = torch.cat((img_inds_gt, bbox2d_gt), dim=1)
            label_gt = kwargs['label_gt'][gt]
            score_gt = label_gt.new_ones(size=label_gt.shape, dtype=torch.float32)
            iou_gt = label_gt.new_ones(size=label_gt.shape, dtype=torch.float32)
            col_gt = [kwargs[i][gt] if i != 'iou' else iou_gt for i in collect]
            roi_num = cfg.roi_num - label_gt.shape[0]
            '''
            bbox2d = torch.where(gt.unsqueeze(1), bbox2d, kwargs['bbox2d_gt'])
            iou_pred = torch.where(gt.unsqueeze(1), iou_pred, iou_pred.new_ones((1,)))
            # iou_heatmap = torch.where(gt, kwargs['iou'], kwargs['iou'].new_ones((1,)))
            # roi_num = cfg.roi_num
            # score = score * iou_heatmap
            score = score * iou_pred.squeeze(1)
        else:
            score = score * iou_pred.squeeze(1)
        # roi_num = cfg.roi_num
        # if self.iou_branch:
        #    score = score * iou_pred.squeeze(1)#.clamp_min(0)

        w = bbox2d[:, 2] - bbox2d[:, 0]
        h = bbox2d[:, 3] - bbox2d[:, 1]
        valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)
        score = score * valid_mask
        '''
        # rois_new = torch.cat((rois[:, 0].unsqueeze(1), bbox2d), dim=1)
        if cfg.min_bbox_size > 0 or cfg.min_score > 0:
            w = bbox2d[:, 2] - bbox2d[:, 0]
            h = bbox2d[:, 3] - bbox2d[:, 1]
            valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size) & (score >= cfg.min_score)
            #if not valid_mask.all():
            bbox2d = bbox2d[valid_mask, :]
            score = score[valid_mask]
            label = label[valid_mask]
            img_inds = img_inds[valid_mask]
            for i in range(len(merge)):
                merge[i] = merge[i][valid_mask]
        
        if bbox2d.numel() > 0:
            if hasattr(cfg, 'nms'):
                cfg.nms.max_num = roi_num
                bbox2d, keep = batched_nms(bbox2d, score, label + img_inds * self.num_classes, cfg.nms)
                bbox2d, score = torch.split(bbox2d, (4, 1), dim=1)
                score = score.squeeze(1)
                label = label[keep]
                img_inds = img_inds[keep]
                for i in range(len(merge)):
                    merge[i] = merge[i][keep]
            elif roi_num < score.shape[0]:
                score, keep = torch.topk(score, roi_num, dim=0)
                bbox2d = bbox2d[keep, :]
                label = label[keep]
                img_inds = img_inds[keep]
                for i in range(len(merge)):
                    merge[i] = merge[i][keep]
        '''
        rois = torch.cat((img_inds.unsqueeze(1), bbox2d), dim=1)
        '''
        if add_gt:
            rois = torch.cat((roi_gt, rois), dim=0)
            label = torch.cat((label_gt, label), dim=0)
            score = torch.cat((score_gt, score), dim=0)
            merge = [torch.cat((i, j), dim=0) for i, j in zip(col_gt, merge)]
            # rois_debug = rois.detach().cpu().numpy()
            # label_debug = label.detach().cpu().numpy()
            # score_debug = score.detach().cpu().numpy()
            # merge_debug = [i.detach().cpu().numpy() for i in merge]
        '''
        return [rois, label, score, iou_pred.squeeze(1), *merge]

    def onnx_export(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    cfg=None,
                    **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed.
                Has shape (B, num_boxes, 5)
            cls_score (Tensor): Box scores. has shape
                (B, num_boxes, num_classes + 1), 1 represent the background.
            bbox_pred (Tensor, optional): Box energies / deltas for,
                has shape (B, num_boxes, num_classes * 4) when.
            img_shape (torch.Tensor): Shape of image.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """

        assert rois.ndim == 3, 'Only support export two stage ' \
                               'model to ONNX ' \
                               'with batch dimension. '

        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        batch_size = scores.shape[0]
        # ignore background class
        scores = scores[..., :self.num_classes]
        labels = torch.arange(
            self.num_classes, dtype=torch.long).to(scores.device)
        labels = labels.view(1, 1, -1).expand_as(scores)
        labels = labels.reshape(batch_size, -1)
        scores = scores.reshape(batch_size, -1)
        bboxes = bboxes.reshape(batch_size, -1, 4)
        if self.reg_class_agnostic:
            bboxes = bboxes.repeat(1, self.num_classes, 1)

        max_size = torch.max(img_shape)
        # Offset bboxes of each class so that bboxes of different labels
        #  do not overlap.
        offsets = (labels * max_size + 1).unsqueeze(2)
        bboxes_for_nms = bboxes + offsets
        max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class',
                                                 cfg.max_per_img)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)
        batch_dets, labels = add_dummy_nms_for_onnx(
            bboxes_for_nms,
            scores.unsqueeze(2),
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            pre_top_k=nms_pre,
            after_top_k=cfg.max_per_img,
            labels=labels)
        # Offset the bboxes back after dummy nms.
        offsets = (labels * max_size + 1).unsqueeze(2)
        # Indexing + inplace operation fails with dynamic shape in ONNX
        # original style: batch_dets[..., :4] -= offsets
        bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
        bboxes -= offsets
        batch_dets = torch.cat([bboxes, scores], dim=2)
        return batch_dets, labels
