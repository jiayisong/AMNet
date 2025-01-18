import math
from ..roi_heads.bbox_heads.bbox2d_head import ConvPool, SAM, TorchInitConvModule
import torch
from mmcv.cnn import xavier_init
import torch.nn as nn
from torch.nn import init
from mmcv.cnn.utils import kaiming_init
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F
from mmcv.ops import DeformUpSample
from mmdet.models.builder import NECKS
from ..backbones.dla import BasicBlock, Root


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=2, affine=True, eps=1e-5):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones([1, self.n_dims, 1, 1]) * scale)
        else:
            self.weight = scale
        self.eps = eps

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = torch.norm(x, dim=[1, ], keepdim=True) + self.eps
        # norm = torch.mean(x ** 2, dim=[1, 2, 3], keepdim=True).sqrt() + self.eps
        x_float = self.weight * x_float
        x_float = x_float / norm
        return x_float.type_as(x)


class ChannelWeight(nn.Module):

    def __init__(self, n_dims, scale=1.0, ):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(ChannelWeight, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.ones([1, self.n_dims, 1, 1]) * scale)

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        x_float = self.weight * x_float
        return x_float.type_as(x)


def norm(x):
    # n = torch.mean(x ** 2, dim=[1, 2, 3], keepdim=True).sqrt() + 1e-6
    # n = torch.norm(x, dim=[2, 3], keepdim=True) + 1e-6
    return x * 50 / (torch.norm(x, dim=[2, 3], keepdim=True) + 1e-6)
    # return x / (torch.mean(x, dim=[1, 2, 3], keepdim=True) + 1e-6)


@NECKS.register_module()
class DLANeck2(BaseModule):
    def __init__(self, in_channels=[32, 64, 128, 256, 512],  # 448, 896, 1280
                 norm_cfg=dict(type='BN', requires_grad=True), init_cfg=None,
                 # conv_cfg=dict(type='DCNv2'),
                 # conv_cfg=dict(type='DCNv2Deterministic'),
                 # conv_cfg=dict(type='DCNv2Fast'),
                 conv_cfg=dict(type='DCNv2Fastv2'),
                 # conv_cfg=None,
                 ):
        super(DLANeck2, self).__init__(init_cfg)
        # self.neck1 = DLATreeDown(in_channels[2:], norm_cfg, conv_cfg)
        self.neck2 = DLATreeUp(in_channels, norm_cfg, conv_cfg)
        # self.neck2 = DLATreeUpSplit(in_channels[2:], norm_cfg, conv_cfg)

    @auto_fp16()
    # @profile
    def forward(self, x):
        x = self.neck2(x)
        return [x[0], ]


class DLATreeUp(BaseModule):
    def __init__(self, in_channels, norm_cfg, conv_cfg):
        super(DLATreeUp, self).__init__()
        repeat = 1
        # self.fuse345_3 = nn.ModuleList([FuseFeature(in_channels[2:], norm_cfg, conv_cfg) for _ in range(repeat)])
        self.fuse34_3 = nn.ModuleList(
            [FuseFeature(in_channels[2:4], in_channels[2], 2, norm_cfg, conv_cfg) for _ in range(2)])
        self.fuse45_4 = nn.ModuleList(
            [FuseFeature(in_channels[3:], in_channels[3], 2, norm_cfg, conv_cfg, ) for _ in range(1)])
        # self.fuse55_5 = nn.ModuleList(
        #    [nn.Sequential(TorchInitConvModule(in_channels[-1], in_channels[-1], 3, 1, 1)) for _ in range(2)])
        # self.C6 = nn.Sequential(
        #     nn.Linear(in_channels[-1], in_channels[-1] * 2),
        #     nn.ReLU(True),
        #     nn.Linear(in_channels[-1] * 2, in_channels[-1]),
        #     nn.ReLU(True),
        # )
        self.fuse5_5 = nn.Sequential(
            # nn.BatchNorm2d(in_channels[-1]),
            # L2Norm(in_channels[-1]),
            # TorchInitConvModule( in_channels[-1], in_channels[-1], 3, 1, 1, norm_cfg=norm_cfg),
            # build_norm_layer(norm_cfg, in_channels[-1])[1], nn.ReLU(True)
        )
        self.fuse4_4 = nn.Sequential(
            # nn.BatchNorm2d(in_channels[-2]),
            # L2Norm(in_channels[-2]),
            # TorchInitConvModule(in_channels[-2], in_channels[-2], 3, 1, 1, norm_cfg=norm_cfg),
            # build_norm_layer(norm_cfg, in_channels[-2])[1], nn.ReLU(True)
        )
        self.fuse3_3 = nn.Sequential(
            # nn.BatchNorm2d(in_channels[-3]),
            # L2Norm(in_channels[-3]),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg),
            # build_norm_layer(norm_cfg, in_channels[-3])[1], nn.ReLU(True)
        )
        # self.fuse = TorchInitConvModule(in_channels[-1] + in_channels[-2] + in_channels[-3], in_channels[-3], 1, 1, 0)
        # self.fuse55_5 = nn.ModuleList([FuseFeature(out_channels[-1:]*2, 1, norm_cfg, conv_cfg) for _ in range(1)])
        # self.fuse35_3 = nn.ModuleList([FuseFeature(in_channels[2:4], in_channels[2], 2, norm_cfg, conv_cfg) for _ in range(1)])

    @auto_fp16()
    # @profile
    def forward(self, x):
        # print(self.weight)
        C5 = x[-1]
        C4 = x[-2]
        C3 = x[-3]

        # C3 = 10 * C3
        # C4 = 10 * C4
        # C5 = 0.1 * C5
        C5 = self.fuse5_5(C5)
        C4 = self.fuse4_4(C4)
        C3 = self.fuse3_3(C3)
        # C5 = norm(C5)
        # C4 = norm(C4)
        # C3 = norm(C3)
        # C6 = self.C6(torch.mean(C5, dim=[2,3], keepdim=False)).unsqueeze(2).unsqueeze(2)
        # C56 = C6 + C5
        # CC5 = self.fuse55_5[0]([C5, C5])
        # C55 = self.fuse55_5[0](C5)
        C45, CC5 = self.fuse45_4[0]([C4, C5])
        C34, CC4 = self.fuse34_3[0]([C3, C4])
        # C345, CC45 = self.fuse34_3[0]([C3, C45])
        # CCC5 = CC5
        # CCC4 = self.fuse45_4[1]([CC4, CC5])
        C3445, CC45 = self.fuse34_3[1]([C34, C45])
        # C4556, CC55 = self.fuse45_4[1]([C45, C56])
        # C4555, CC55 = self.fuse45_4[1]([C45, C55])
        # C5555 = self.fuse55_5[1](C55)
        # C34445556, CC4556 = self.fuse34_3[2]([C3445, C4556])
        # C34445555, CC4555 = self.fuse34_3[2]([C3445, C4555])
        # C34455, CCC5 = self.fuse35_3[0]([C3445, CC5])
        # CCCC5 = CCC5
        # CCCC4 = self.fuse45_4[1]([CC4, CC5])
        # CCCC3 = self.fuse34_3[2]([CCC3, CCC4])
        # CC3 = C3 + F.interpolate(CC4, scale_factor=2)
        # C = self.fuse(torch.cat([C3445, F.interpolate(C4555, scale_factor=2), F.interpolate(C5555, scale_factor=4)], 1))
        return (C3445,)


class FuseFeature(BaseModule):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_cfg=dict(type='BN', requires_grad=True),
                 conv_cfg=dict(type='myDCNv2')):
        super(FuseFeature, self).__init__()
        kernel = 3
        self.scale_factor = scale_factor
        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(in_channels[1]),
            #TorchInitConvModule(in_channels[1], out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=None),
            # SE(in_channels[1], out_channels, 1, norm_cfg),
            # AddCood2(h, w),
            # nn.GroupNorm(64, in_channels[1]),
            # ChannelWeight(in_channels[1]),
            ConvModule(in_channels[1], out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(in_channels[1], out_channels, 1, 1, 0, norm_cfg=norm_cfg),
            # MyAct(),
            # TorchInitConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg),
            # AddCood2(h, w),
            # TorchInitConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg),
            # ConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),

            # TorchInitConvModule(in_channels[1], out_channels, 3, 1, 1, norm_cfg=norm_cfg),
            # TorchInitConvModule(in_channels[1], out_channels, 1, 1, 0, norm_cfg=norm_cfg),
            # TorchInitConvModule(in_channels[1] * 2, out_channels, 1, 1, 0, norm_cfg=norm_cfg),
            # TorchInitConvModule(in_channels[1], out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
            # TorchInitConvModule(out_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=None),
            # TorchInitConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
            # TorchInitConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
            # TorchInitConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
            # MyAct(),
            # TorchInitConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=None),
            # ConvModule(out_channels // 2, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg, ),
            # TorchInitConvModule(i, out_channel, 1, 1, 0, norm_cfg=norm_cfg),
            # ResBlock(in_channels[1], out_channels, kernel, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ResBlock(out_channel, kernel, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ResBlock(out_channel, kernel, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # SPP(i, out_channel, norm_cfg, (5, 9, 13, 17)),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            nn.UpsamplingNearest2d(scale_factor=scale_factor),
            # nn.PixelShuffle(scale_factor),
            # DeformUpSample(out_channels, scale_factor=scale_factor, kernel_size=1),
            # TorchInitConvModule(out_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg),
            # DeformTransConv(i, out_channel, scale_factor, kernel, norm_cfg, conv_cfg)
            # nn.UpsamplingBilinear2d(scale_factor=2)
        )
        # '''
        self.conv0 = nn.Sequential(
            #ChannelWeight(in_channels[0]),
            # TorchInitConvModule(in_channels[0], out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=None),
            # TorchInitConvModule(in_channels[0], out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
        )
        # '''
        # self.add = AdoptAdd3(out_channels)
        self.root = nn.Sequential(
            # ConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # SE(out_channels),
            # TorchInitConvModule(out_channels, out_channels, 7, 1, 3, groups=out_channels, norm_cfg=norm_cfg),
            # TorchInitConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg),
            # ConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg,conv_cfg=conv_cfg),
            # ConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # SE(out_channels),
            # TorchInitConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg),
            # ConvModule(out_channels, out_channels // 2, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg, ),
            # ConvModule(out_channels // 2, out_channels // 2, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg, ),
            # SPP(out_channel, out_channel, norm_cfg, (5, 9, 13, 17)),
            # ConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(out_channel, out_channel, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
        )
        # self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(1, in_channels, 1, 1)) for _ in range(fuse_num)])
        # self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for _ in range(len(in_channels) + 1)])
        # for i in self.weight:
        #    nn.init.constant_(i, 0.5)
        # self.bias = nn.Parameter(torch.Tensor(1, 1, 1, 1))
        # for i in self.weight:
        #     nn.init.constant_(i, 1)
        # nn.init.constant_(self.bias, 0)

    @auto_fp16()
    # @profile
    def forward(self, x):
        x0, x1 = x
        # x11 = self.conv1[0](x1)
        # x12 = self.conv1[1](x11)
        # x13 = self.conv1[2](x12)
        # x14 = self.conv1[3](x13)
        # x1 = torch.cat([x11, x12, x13, x14], 1)
        x1 = self.conv1(x1)
        # x1 = self.conv1[1](x11)  # + x11
        # x0, x1 = self.add([x0, x1])
        # x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='nearest')
        y = x1
        # x1 = self.add([x0, x1])
        if x0 is not None:
            x1 = x1 + self.conv0(x0)
        # x1 = torch.cat([x1, x0], 1)
        #x1 = F.relu(x1, inplace=True)
        # x1 = torch.maximum(x0, x1)
        # x1 = x1 * x0
        x = self.root(x1)
        # x = x * 0.5
        return x, y


class DLABlock(BaseModule):
    def __init__(self, in_channels, depth, norm_cfg):
        super(DLABlock, self).__init__()
        block = []
        root = []

        for i in range(depth):
            block.append(BasicBlock(in_channels, in_channels, norm_cfg=norm_cfg, conv_cfg=None))

            if i % 2 == 1:
                n = self.factor2_num(i + 1) + 1
                root.append(Root(in_channels * n, in_channels, norm_cfg=norm_cfg, conv_cfg=None, kernel_size=1,
                                 add_identity=False))
        self.block = nn.ModuleList(block)
        self.root = nn.ModuleList(root)

    def factor2_num(self, n):
        a = 0
        while n % 2 == 0 and n > 1:
            a += 1
            n = n // 2
        return a

    def forward(self, x):
        y = []
        num = []
        for i, b in enumerate(self.block):
            x = b(x)
            y.append(x)
            num.append(1)
            if i % 2 == 1:
                # print(num)
                r = [y[-1]]
                y.pop(-1)
                n = 0
                while len(num) > 1 and num[-1] == num[-2]:
                    r.append(y[-1])
                    y.pop(-1)
                    num[-2] = num[-2] * 2
                    num.pop(-1)
                n += num.pop(-1)
                x = self.root[(i - 1) // 2](r)
                y.append(x)
                num.append(n)
                # print(num)
        return x


class SPP(BaseModule):
    def __init__(self, in_channel, out_channel, norm_cfg, k=(3, 5, 7, 9, 11)):
        super(SPP, self).__init__()
        self.pool = nn.ModuleList()
        for i in k:
            self.pool.append(nn.MaxPool2d(i, 1, i // 2))
        self.conv = TorchInitConvModule(in_channel * (len(k) + 1), out_channel, 1, 1, 0, norm_cfg=norm_cfg)
        self.attention = nn.Sequential(nn.Conv2d(in_channel, (len(k) + 1), 3, 1, 1),
                                       nn.Sigmoid())

    def forward(self, x):
        atten = self.attention(x)
        atten = torch.split(atten, 1, 1)
        y = [x * atten[0]]
        for i, a in zip(self.pool, atten[1:]):
            y.append(i(x) * a)
        y = torch.cat(y, dim=1)
        y = self.conv(y)
        return y


class SE(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, norm_cfg):
        super(SE, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, out_channels),
                                # nn.ReLU(True),
                                # nn.Linear(in_channels, in_channels),
                                nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        init.orthogonal_(self.fc[0].weight, gain=1 / math.sqrt(3))
        init.orthogonal_(self.conv.weight, gain=1 / math.sqrt(3))
        # init.orthogonal_(self.fc[2].weight, gain=1 / math.sqrt(3))
        init.constant_(self.fc[0].bias, 0)
        init.constant_(self.conv.bias, 0)
        # init.constant_(self.fc[2].bias, 0)

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=False)
        x = self.conv(x) * self.fc(y).unsqueeze(2).unsqueeze(2)
        x = self.relu(self.bn(x))
        return x


class AdoptAdd(BaseModule):
    def __init__(self, in_channels):
        super(AdoptAdd, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels * 2, in_channels * 2),
                                nn.ReLU(True),
                                nn.Linear(in_channels * 2, in_channels * 2),
                                nn.Sigmoid())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        init.orthogonal_(self.fc[0].weight, gain=1 / math.sqrt(3))
        init.orthogonal_(self.fc[2].weight, gain=1 / math.sqrt(3))
        init.constant_(self.fc[0].bias, 0)
        init.constant_(self.fc[2].bias, 0)

    def forward(self, x):
        x1, x2 = x
        y1 = torch.mean(x1, dim=[2, 3], keepdim=False)
        y2 = torch.mean(x2, dim=[2, 3], keepdim=False)
        y1, y2 = self.fc(torch.cat([y1, y2], 1)).unsqueeze(2).unsqueeze(2).chunk(2, dim=1)
        x = [x1 * y1, x2 * y2]
        return x


class AdoptAdd2(BaseModule):
    def __init__(self, in_channels):
        super(AdoptAdd2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels * 2, 2, 3, 1, 1),
                                  nn.Sigmoid())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        init.orthogonal_(self.conv[0].weight, gain=1 / math.sqrt(3))
        # init.orthogonal_(self.fc[2].weight, gain=1 / math.sqrt(3))
        init.constant_(self.conv[0].bias, 0)
        # init.constant_(self.fc[2].bias, 0)

    def forward(self, x):
        x1, x2 = x
        y1, y2 = self.conv(torch.cat([x1, x2], 1)).chunk(2, dim=1)
        x = x1 * y1 + x2 * y2
        return x


class AdoptAdd3(BaseModule):
    def __init__(self, in_channels):
        super(AdoptAdd3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                  nn.Sigmoid())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        init.orthogonal_(self.conv[0].weight, gain=1 / math.sqrt(3))
        # init.orthogonal_(self.fc[2].weight, gain=1 / math.sqrt(3))
        init.constant_(self.conv[0].bias, 0)
        # init.constant_(self.fc[2].bias, 0)

    def forward(self, x):
        x1, x2 = x
        y1, y2 = self.conv(torch.cat([x1, x2], 1)).chunk(2, dim=1)
        x = x1 * y1 + x2 * y2
        return x


class SA(BaseModule):
    def __init__(self):
        super(SA, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 5, 1, 2),
                                  nn.ReLU(True),
                                  nn.Conv2d(1, 1, 5, 1, 2),
                                  nn.Sigmoid())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        init.orthogonal_(self.conv[0].weight, gain=1 / math.sqrt(3))
        init.orthogonal_(self.conv[2].weight, gain=1 / math.sqrt(3))
        init.constant_(self.conv[0].bias, 0)
        init.constant_(self.conv[2].bias, 0)

    def forward(self, x):
        y = torch.mean(x, dim=[1], keepdim=True)
        x = x * self.conv(y)
        return x


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)


class ResBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, norm_cfg, conv_cfg):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size, 1, kernel_size // 2, norm_cfg=norm_cfg,
                       conv_cfg=conv_cfg),
            ConvModule(out_channels, out_channels, kernel_size, 1, kernel_size // 2, norm_cfg=norm_cfg,
                       conv_cfg=conv_cfg, act_cfg=None),
        )
        self.short_cut = nn.Sequential() if in_channels == out_channels else \
            TorchInitConvModule(in_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        x = F.relu(self.conv(x) + self.short_cut(x), True)
        return x


class MyUpSample(BaseModule):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, norm_cfg, conv_cfg):
        super(MyUpSample, self).__init__()
        self.conv = nn.ModuleList([
            ConvModule(in_channels, out_channels, kernel_size, 1, kernel_size // 2, norm_cfg=norm_cfg,
                       conv_cfg=conv_cfg)
            for i in range(scale_factor ** 2)]
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        x = torch.stack([i(x) for i in self.conv], 2)
        x = F.pixel_shuffle(x.view(x.shape[0], -1, x.shape[3], x.shape[4]), self.scale_factor)
        return x


class AddCood(BaseModule):
    def __init__(self):
        super(AddCood, self).__init__()
        self.gamma = nn.Parameter(torch.zeros([1, 2, 1, 1]))
        self.beta = nn.Parameter(torch.ones([1, 2, 1, 1]))

    def forward(self, x):
        b, c, h, w = x.shape
        gx = torch.arange(w, device=x.device, dtype=x.dtype) / (w - 1.) * 2
        gy = torch.arange(h, device=x.device, dtype=x.dtype) / (h - 1.) * 2
        gx = gx.view([1, 1, 1, w]).expand([b, 1, h, w])
        gy = gy.view([1, 1, h, 1]).expand([b, 1, h, w])
        gxgy = torch.cat([gx, gy], 1) * self.beta + self.gamma
        x = torch.cat([x, gxgy], 1)
        return x


class AddCood2(BaseModule):
    def __init__(self, h, w):
        super(AddCood2, self).__init__()
        self.xy = nn.Parameter(torch.zeros([1, 2, h, w]))
        gx = torch.arange(w) / (w - 1.) * 2
        gy = torch.arange(h) / (h - 1.) * 2
        gx = gx.view([1, 1, 1, w]).expand([1, 1, h, w])
        gy = gy.view([1, 1, h, 1]).expand([1, 1, h, w])
        gxgy = torch.cat([gx, gy], 1)
        self.xy.data = gxgy[:]
        # print(self.xy)

    def forward(self, x):
        x = torch.cat([x, self.xy.expand(x.shape[0], 2, x.shape[2], x.shape[3])], 1)
        return x

class AddCood3(BaseModule):
    def __init__(self, h, w):
        super(AddCood3, self).__init__()
        self.xy = nn.Parameter(torch.ones([1, 2, h, w]))


    def forward(self, x):
        x = torch.cat([x, self.xy.expand(x.shape[0], 2, x.shape[2], x.shape[3])], 1)
        return x

class MyAct(BaseModule):
    def __init__(self):
        super(MyAct, self).__init__()

    def forward(self, x):
        x = torch.cat([F.relu(x), -F.relu(-x)], 1)
        return x
