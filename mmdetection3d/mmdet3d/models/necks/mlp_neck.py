import math

import torch
from mmcv.cnn import xavier_init
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2dPack, ModulatedDeformConv2d, modulated_deform_conv2d
from mmdet.models.builder import NECKS
from ..backbones.dla import dla_build_norm_layer


@NECKS.register_module()
class MLPNeck(BaseModule):
    def __init__(self, in_channels=[16, 32, 64, 128, 256, 512], out_channels=128,
                 norm_cfg=dict(type='BN', requires_grad=True), conv_cfg=dict(type='DCNv2'),
                 init_cfg=None):
        super(MLPNeck, self).__init__(init_cfg)
        self.up1 = nn.Sequential(ConvModule(in_channels[-1], out_channels, 3, 1, 1, norm_cfg=norm_cfg,
                                            conv_cfg=dict(type='myDCNv2')),
                                 nn.UpsamplingNearest2d(scale_factor=4))
        self.up2 = nn.Sequential(ConvModule(in_channels[-2], out_channels, 3, 1, 1, norm_cfg=norm_cfg,
                                            conv_cfg=dict(type='myDCNv2')),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.up3 = nn.Sequential(ConvModule(in_channels[-3], out_channels, 3, 1, 1, norm_cfg=norm_cfg,
                                            conv_cfg=dict(type='myDCNv2')))
        self.mlp = nn.Sequential(FeatureLinear(3, 2, out_channels, norm_cfg),
                                 FeatureLinear(2, 1, out_channels, norm_cfg),
                                 )

    @auto_fp16()
    # @profile
    def forward(self, x):
        C5 = x[-1]
        C4 = x[-2]
        C3 = x[-3]

        P5 = self.up1(C5)
        P4 = self.up2(C4)
        P3 = self.up3(C3)

        P = self.mlp([P5, P3, P4])
        return P


class FeatureLinear(BaseModule):
    def __init__(self, in_features, out_features, out_channels, norm_cfg):
        super(FeatureLinear, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_features, out_features),
                                    nn.ReLU(inplace=True))
        self.conv = nn.ModuleList([ConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg,
                                              conv_cfg=dict(type='myDCNv2')) for _ in range(out_features)])

    def forward(self, input):
        P = torch.stack(input, dim=-1)
        P = self.linear(P)
        P = torch.split(P, 1, -1)
        P = [self.conv[i](p.squeeze(-1)) for i, p in enumerate(P)]
        return P
