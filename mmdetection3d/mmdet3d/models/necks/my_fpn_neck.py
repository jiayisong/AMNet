import torch
from mmcv.cnn import xavier_init
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2dPack, ModulatedDeformConv2d, modulated_deform_conv2d
from mmdet.models.builder import NECKS


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
                build_conv_layer(conv_cfg, in_channels, out_channels, 1, stride=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1])
            self.conv1 = nn.Sequential(nn.PixelUnshuffle(stride),
                                       build_conv_layer(
                                           conv_cfg, in_channels * stride * stride, out_channels, 1, stride=1,
                                           padding=0, bias=False))
        elif stride < 1:
            self.identity = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=1 / stride),
                build_conv_layer(conv_cfg, in_channels, out_channels, 1, stride=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1])
            self.conv1 = nn.Sequential(build_conv_layer(
                conv_cfg, in_channels, out_channels * int(1 / stride) * int(1 / stride), 1, stride=1,
                padding=0, bias=False),
                nn.PixelShuffle(int(1 / stride)), )
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

        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]
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
class MyFPNNeck(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, in_channels=[16, 32, 64, 128, 256, 512], out_channels=512,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super(MyFPNNeck, self).__init__(init_cfg)
        # self.up5 = DeformUpSample(in_channels[-1], in_channels[-3], 4, norm_cfg, kernel_size=1)
        # self.up4 = DeformUpSample(in_channels[-2], in_channels[-3], 2, norm_cfg, kernel_size=1)
        # self.up3 = DeformUpSample(in_channels[-3], in_channels[-3], 1, norm_cfg, kernel_size=1)
        self.conv = nn.Sequential(
            ConvModule(
                in_channels[-1] + in_channels[-2] + in_channels[-3],
                # in_channels[-3] * 3,
                out_channels, 1, padding=0, bias=False, norm_cfg=norm_cfg,  # deform_groups=4,
                # conv_cfg=dict(type='DCNv2'),
            ),
            #BasicBlock(out_channels, out_channels, norm_cfg, conv_cfg=dict(type='DCNv2'), ),
            #BasicBlock(out_channels, out_channels, norm_cfg, conv_cfg=dict(type='DCNv2'), ),
            #BasicBlock(out_channels, out_channels, norm_cfg, conv_cfg=dict(type='DCNv2'), ),
        )
        '''
        self.p5 = ConvModule(in_channels[-1], in_channels[-2], 3, padding=1, bias=False,  # deform_groups=4,
                             conv_cfg=dict(type='DCNv2'), norm_cfg=norm_cfg)
        self.up5 = build_conv_layer(dict(type='deconv'), in_channels[-2], in_channels[-2], 4,
                                    stride=2, padding=1, output_padding=0, groups=in_channels[-2], bias=False)
        self.p4 = ConvModule(in_channels[-2], in_channels[-3], 3, padding=1, bias=False,  # deform_groups=4,
                             conv_cfg=dict(type='DCNv2'), norm_cfg=norm_cfg)
        self.up4 = build_conv_layer(dict(type='deconv'), in_channels[-3], in_channels[-3], 4,
                                    stride=2, padding=1, output_padding=0, groups=in_channels[-3], bias=False)
        self.p3 = ConvModule(in_channels[-3], in_channels[-3], 3, padding=1, bias=False,  # deform_groups=4,
                             conv_cfg=dict(type='DCNv2'), norm_cfg=norm_cfg)
        '''

    def init_weights(self):
        for m in self.conv[0].modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    @auto_fp16()
    # @profile
    def forward(self, x):
        C5 = x[-1]
        C4 = x[-2]
        C3 = x[-3]
        '''
        # C5 = self.p5(C5)
        P5 = self.up5(C5)
        P5 = P5 + C4
        # P5 = self.p4(P5)
        P4 = self.up4(P5)
        P4 = P4 + C3
        # P3 = self.up3(P4)
        P3 = P4
        '''
        '''
        P5 = C5
        P5_up = F.pixel_shuffle(P5, 2)
        #P5_up = F.upsample_nearest(P5, scale_factor=2)
        P4 = torch.cat((C4, P5_up), 1)
        P4_up = F.pixel_shuffle(P4, 2)
        #P4_up = F.upsample_nearest(P4, scale_factor=2)
        P3 = torch.cat((C3, P4_up), 1)
        #P3_up = F.pixel_shuffle(P3, 2)
        #P2 = torch.cat((C2, P3_up), 1)
        #P2_up = P2
        '''
        P5 = F.interpolate(C5, scale_factor=4)
        P4 = F.interpolate(C4, scale_factor=2)
        P3 = C3
        # P5 = self.up5(C5)
        # P4 = self.up4(C4)
        # P3 = self.up3(C3)
        out = torch.cat((P3, P4, P5), 1)
        out = self.conv(out)
        # print(f'neck_bn_conv {self.conv[0].norm.weight[0].item():.6f} {self.conv[0].conv.weight[0, 0, 0, 0].item():.6f}')
        return (out,)  # , P4, P5


class D2S(nn.Module):
    def __init__(self, in_channel, scale_factor):
        super(D2S, self).__init__()
        self.scale_factor_sqaure = scale_factor ** 2
        assert (in_channel % self.scale_factor_sqaure) == 0
        self.scale_factor = scale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c // self.scale_factor_sqaure, self.scale_factor, self.scale_factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, c // self.scale_factor_sqaure, h * self.scale_factor, w * self.scale_factor)
        return x


'''
class ModulatedDeformConv2dPack2(ModulatedDeformConv2dPack):

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        print('offset', offset)
        mask = torch.sigmoid(mask)
        print('mask', mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

'''


class DeformUpSample(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor, norm_cfg, kernel_size=3):
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
