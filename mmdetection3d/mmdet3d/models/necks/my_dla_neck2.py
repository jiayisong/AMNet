import torch
from mmcv.cnn import xavier_init
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2dPack, ModulatedDeformConv2d, modulated_deform_conv2d
from mmdet.models.builder import NECKS
from ..backbones.dla import dla_build_norm_layer


class Root(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 conv_cfg,
                 kernel_size,
                 init_cfg=None):
        super(Root, self).__init__(init_cfg)
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False)
        self.bn = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_list):
        """Forward function.

        Args:
            feat_list (list[torch.Tensor]): Output features from
                multiple layers.
        """
        x = self.conv(torch.cat(feat_list, 1))
        x = self.bn(x)
        x = self.relu(x)
        return x



class BasicBlock(BaseModule):
    """BasicBlock in DLANet.

    Args:
        in_channels (int): Input feature channel.
        out_channels (int): Output feature channel.
        norm_cfg (dict): Dictionary to construct and config
            norm layer.
        conv_cfg (dict): Dictionary to construct and config
            conv layer.
        stride (int, optional): Conv stride. Default: 1.
        dilation (int, optional): Conv dilation. Default: 1.
        init_cfg (dict, optional): Initialization config.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 conv_cfg,
                 stride=1,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        if stride > 1:
            self.identity = nn.Sequential(
                nn.MaxPool2d(stride, stride),
                build_conv_layer(None, in_channels, out_channels, 1, stride=1, bias=False),
                dla_build_norm_layer(norm_cfg, out_channels)[1])
            self.conv1 = nn.Sequential(nn.PixelUnshuffle(stride),
                                       build_conv_layer(
                                           conv_cfg, in_channels * stride * stride, out_channels, 1, stride=1,
                                           padding=0, bias=False))
        elif stride < 1:
            self.identity = nn.Sequential(
                build_conv_layer(None, in_channels, out_channels, 1, stride=1, bias=False),
                dla_build_norm_layer(norm_cfg, out_channels)[1],
                nn.UpsamplingNearest2d(scale_factor=1 / stride),
            )
            self.conv1 = DeformUpSample(in_channels, out_channels, int(1 / stride), norm_cfg)
            # self.conv1 = nn.Sequential(build_conv_layer(conv_cfg, in_channels, out_channels * int(1 / stride) * int(1 / stride), 1, stride=1,padding=0, bias=False), nn.PixelShuffle(int(1 / stride)), )
        else:
            self.identity = None
            self.conv1 = build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                bias=False)

        self.bn1 = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.stride = stride

    def forward(self, x):
        """Forward function."""
        if self.identity is None:
            identity = x
        else:
            identity = self.identity(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

@NECKS.register_module()
class MyDLANeck2(BaseModule):
    def __init__(self, in_channels=[16, 32, 64, 128, 256, 512],
                 norm_cfg=dict(type='BN', requires_grad=True), conv_cfg=dict(type='DCNv2'),
                 init_cfg=None):
        super(MyDLANeck2, self).__init__(init_cfg)
        # self.up1 = DeformUpSample(in_channels[-1], in_channels[-2], 2, norm_cfg)

        self.up1 = nn.Sequential(ConvModule(in_channels[-1], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                            conv_cfg=dict(type='myDCNv2')),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.block1 = ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                 conv_cfg=dict(type='myDCNv2'))
        self.root1 = ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                conv_cfg=dict(type='myDCNv2'))

        self.up2 = nn.Sequential(ConvModule(in_channels[-2], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
                                            conv_cfg=dict(type='myDCNv2')),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.block2 = ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
                                 conv_cfg=dict(type='myDCNv2'))
        self.root2 = ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
                                conv_cfg=dict(type='myDCNv2'))

        self.up3 = nn.Sequential(ConvModule(in_channels[-2], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
                                            conv_cfg=dict(type='myDCNv2')),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.block3 = ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
                                 conv_cfg=dict(type='myDCNv2'))
        self.root3 = ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
                                conv_cfg=dict(type='myDCNv2'))
        '''
        self.up1 = nn.Sequential(BasicBlock(in_channels[-1], in_channels[-2], norm_cfg, conv_cfg, 0.5),
                                 BasicBlock(in_channels[-2], in_channels[-3], norm_cfg, conv_cfg, 0.5),)
        self.up2 = nn.Sequential(BasicBlock(in_channels[-2], in_channels[-3], norm_cfg, conv_cfg, 0.5),
                                 BasicBlock(in_channels[-3], in_channels[-3], norm_cfg, conv_cfg, 1), )
        self.up3 = nn.Sequential(BasicBlock(in_channels[-3], in_channels[-3], norm_cfg, conv_cfg, 1),
                                 BasicBlock(in_channels[-3], in_channels[-3], norm_cfg, conv_cfg, 1), )
        '''
        '''
        self.up1 = nn.Sequential(ConvModule(in_channels[-1], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.conv = nn.Sequential(ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                             conv_cfg=dict(type='myDCNv2')),
                                  ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                             conv_cfg=dict(type='myDCNv2')),
                                  ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                             conv_cfg=dict(type='myDCNv2')),
                                  ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                             conv_cfg=dict(type='myDCNv2')),
                                  ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                             conv_cfg=dict(type='myDCNv2')),
                                  ConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg,
                                             conv_cfg=dict(type='myDCNv2')),
                                  )
        
        '''

    @auto_fp16()
    # @profile
    def forward(self, x):
        C5 = x[-1]
        C4 = x[-2]
        C3 = x[-3]

        root1 = self.up1(C5)
        root2 = self.block1(C4)
        C5_up = self.root1(root1 + root2)
        # root2 = C4
        # C5_up = self.root1(root1 + root2)

        root1 = self.up2(C4)
        root2 = self.block2(C3)
        C4_up = self.root2(root1 + root2)
        # root2 = C3
        # C4_up = self.root2(root1 + root2)

        root1 = self.up3(C5_up)
        root2 = self.block3(C4_up)
        C5_up_up = self.root3(root1 + root2)
        # root2 = C4_up
        # C5_up_up = self.root3(root1 + root2)

        return (C5_up_up,)  # , P4, P5
        '''
        #y = self.conv(self.up1(C5) + C4)
        y = self.up1(C5) + self.up2(C4) + self.up3(C3)
        return (y,)
        '''

'''
class DeformUpSample(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor, norm_cfg, kernel_size=1):
        super(DeformUpSample, self).__init__()
        self.scale_factor_sqaure = scale_factor ** 2
        self.scale_factor = scale_factor
        self.out_channel = out_channel
        self.dcn = ModulatedDeformConv2dPack(in_channels=in_channel * self.scale_factor_sqaure,
                                             out_channels=out_channel * self.scale_factor_sqaure, stride=1,
                                             kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
                                             groups=self.scale_factor_sqaure,
                                             deform_groups=self.scale_factor_sqaure)
        self.bnrelu = nn.Sequential(build_norm_layer(norm_cfg, out_channel)[1], nn.ReLU(inplace=True))
        self.dcn.weight.data.fill_(1)
    # @profile
    def forward(self, x):
        b, c, h, w = x.size()
        # x = x.unsqueeze(1).expand([b, self.scale_factor_sqaure, c, h, w]).reshape([b, self.scale_factor_sqaure * c, h, w])
        x = x.repeat(1, self.scale_factor_sqaure, 1, 1)
        x = self.dcn(x)
        x = x.view(b, self.scale_factor, self.scale_factor, self.out_channel, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2)
        x = x.reshape(b, self.out_channel, h * self.scale_factor, w * self.scale_factor)
        x = self.bnrelu(x)
        return x
'''


class DeformUpSample(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor, norm_cfg, kernel_size=1):
        super(DeformUpSample, self).__init__()
        self.scale_factor_sqaure = scale_factor ** 2
        self.scale_factor = scale_factor
        self.out_channel = out_channel
        self.dcn = nn.ModuleList()
        for _ in range(self.scale_factor_sqaure):
            self.dcn.append(ModulatedDeformConv2dPack(in_channels=in_channel,
                                                      out_channels=out_channel, stride=1,
                                                      kernel_size=kernel_size, padding=kernel_size // 2, bias=False))
        self.bnrelu = nn.Sequential(build_norm_layer(norm_cfg, out_channel)[1], nn.ReLU(inplace=True))

    # @profile
    def forward(self, x):
        b, c, h, w = x.size()
        y = []
        for i in self.dcn:
            y.append(i(x))
        x = torch.cat(y, dim=1)
        x = x.view(b, self.scale_factor, self.scale_factor, self.out_channel, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2)
        x = x.reshape(b, self.out_channel, h * self.scale_factor, w * self.scale_factor)
        x = self.bnrelu(x)
        return x


class SPP(nn.Module):
    def __init__(self, k=(3, 5, 7, 9, 11)):
        super(SPP, self).__init__()
        self.pool = nn.ModuleList()
        for i in k:
            self.pool.append(nn.MaxPool2d(i, 1, i // 2))

    def forward(self, x):
        y = [x]
        for i in self.pool:
            y.append(i(x))
        return torch.cat(y, dim=1)
