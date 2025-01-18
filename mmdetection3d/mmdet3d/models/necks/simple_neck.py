import math
from ..roi_heads.bbox_heads.bbox2d_head import ConvPool, SAM, TorchInitConvModule
import torch
from torch.nn import init
from mmcv.cnn import xavier_init
import torch.nn as nn
from mmcv.cnn.utils import kaiming_init
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F
from mmcv.ops import DeformUpSample, DeformTransConv
from mmdet.models.builder import NECKS
from ..backbones.dla import BasicBlock, Root
from ..roi_heads.bbox_heads.bbox2d_head import ConvPool, SAM, TorchInitConvModule


@NECKS.register_module()
class SimpleNeck(BaseModule):
    def __init__(self, in_channels=[32, 64, 128, 256, 512], out_channels=[32, 64, 128, 256, 512],  # 448, 896, 1280
                 norm_cfg=dict(type='BN', requires_grad=True), init_cfg=None,
                 # conv_cfg=dict(type='DCNv2'),
                 # conv_cfg=dict(type='DCNv2Deterministic'),
                 # conv_cfg=dict(type='DCNv2Fast'),
                 conv_cfg=dict(type='DCNv2Fastv2'),
                 upsample='deform'
                 # conv_cfg=None,
                 ):
        super(SimpleNeck, self).__init__(init_cfg)
        '''
        k = 4
        self.neck = nn.Sequential(
            DenseBlock(in_channels[-2], in_channels[-3], norm_cfg, conv_cfg, k),
            TorchInitConvModule(in_channels[-3] * k, in_channels[-3] * 4, 1, 1, 0),
            nn.PixelShuffle(2),
        )
        '''
        self.C5 = nn.Sequential(
            # TorchInitConvModule(in_channels[-1], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg),
            # SE(in_channels[-1]),
            # ResBlock(in_channels[-1], in_channels[-2], norm_cfg, conv_cfg),
            # ResBlock(in_channels[-2], in_channels[-3], norm_cfg, conv_cfg),
            TorchInitConvModule(in_channels[-1], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg),
            ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # DeformTransConv(in_channels[-3], in_channels[-3], 3, 4, 1),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg),
            # ResBlock2(in_channels[-1], in_channels[-3], 2, 3, norm_cfg, conv_cfg),
            # DenseBlock(in_channels[-1], in_channels[-3], 2, 3, norm_cfg, conv_cfg),
            ConvModule(in_channels[-3], in_channels[-3], 3, 4, 1, norm_cfg=norm_cfg, conv_cfg=dict(type='DCNTrans'),
                       #  act_cfg=None,
                       ) if upsample == 'deform' else
            nn.Upsample(scale_factor=4, mode=upsample),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg,
            #                     #act_cfg=None
            #                     ),
            # SA(in_channels[-3]),
            # SE(in_channels[-3]),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, act_cfg=None),
            # nn.UpsamplingNearest2d(scale_factor=4),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 4, 1, norm_cfg=norm_cfg, conv_cfg=dict(type='DCNTrans')),
        )
        self.C4 = nn.Sequential(
            # SE(in_channels[-2]),
            # ResBlock(in_channels[-2], in_channels[-3], norm_cfg, conv_cfg),
            # ConvModule(in_channels[-2], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            TorchInitConvModule(in_channels[-2], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
                               # act_cfg=None,
                                ),
            # TorchInitConvModule(in_channels[-2], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg,
            #                    # act_cfg=None,
            #                     ),
            #TorchInitConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg),
            ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg,
            #            #act_cfg=None,
            #            ),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg),
            # DeformTransConv(in_channels[-3], in_channels[-3], 3, 2, 1),
            # ResBlock2(in_channels[-2], in_channels[-3], 1, 3, norm_cfg, conv_cfg),
            #DenseBlock(in_channels[-2], in_channels[-3], 1, 3, norm_cfg, conv_cfg),
            ConvModule(in_channels[-3], in_channels[-3], 3, 2, 1, norm_cfg=norm_cfg, conv_cfg=dict(type='DCNTrans'),
                        act_cfg=None,
                       ) if upsample == 'deform' else
            nn.Upsample(scale_factor=2, mode=upsample),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg,
            #                     #act_cfg=None
            #                     ),
            # SA(in_channels[-3]),
            # SE(in_channels[-3]),
        )
        self.C45 = nn.Sequential(
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg,
            #                     #act_cfg=None,
            #                     ),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg,
            #           # act_cfg=None,
            #            ),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 2, 1, norm_cfg=norm_cfg, conv_cfg=dict(type='DCNTrans'),
            #          #  act_cfg=None,
            #            ),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),

            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg, act_cfg=None),
            # nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.C3 = nn.Sequential(
            # SE(in_channels[-3]),
            # ChannelWeight(in_channels[-3]),
            # SA(in_channels[-3]),
            TorchInitConvModule(in_channels[-3], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg,
                                  act_cfg=None
                                  ),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg),
        )
        self.root = nn.Sequential(
            # AddCood2(48, 160),
            # nn.ReLU(True),
            # TorchInitConvModule(in_channels[-3] * 3, in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg),
            # ConvModule(in_channels[-3], in_channels[-3], 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # TorchInitConvModule(in_channels[-3], in_channels[-3], 1, 1, 0, norm_cfg=norm_cfg, act_cfg=None),
        )
        self.bnrelu = nn.ModuleList([
            nn.Sequential(
                #build_norm_layer(norm_cfg, in_channels[-3])[1],
                nn.ReLU(True),
            ) for _ in range(2)
        ])
        # self.fuse = SpaceSelect(in_channels[-3], 3)
        #self.reluadd = ReLUAdd(3)
        #self.weightadd = WeightAdd(3)

    @auto_fp16()
    # @profile
    def forward(self, x):
        C5 = x[-1]
        C4 = x[-2]
        C3 = x[-3]
        # C5 = C5 * 0.05
        # x = self.neck(x[-2])
        # C4 = F.upsample_nearest(self.C41(C4) + self.C42(C4), scale_factor=2)
        # C5 = F.upsample_nearest(self.C51(C5) + self.C52(C5), scale_factor=4)
        C3 = self.C3(C3)
        C4 = self.C4(C4)
        C5 = self.C5(C5)
        # x = C3 + self.C45(C4 + C5)
        # x = C3 + C4 + C5
        # x = F.relu(F.relu(C3 + C4, True) + C5, True)
        # x = F.relu(F.relu(C5 + C4, True) + C3, True)
        x = self.bnrelu[1](self.bnrelu[0](C5 + C4) + C3)
        # x = self.bnrelu[1](self.C45(self.bnrelu[0](C4 + C5)) + C3)
        # x = self.reluadd([C3, C4, C5]).
        #x = self.weightadd([C3, C4, C5])
        # x = C3 + self.bnrelu[0](C4) + self.bnrelu[1](C5) + self.bnrelu[2](C4 + C5)
        # x = self.bnrelu[0](C3) + self.bnrelu[1](C4) + self.bnrelu[2](C5) + self.bnrelu[3](C3 + C4 + C5)
        # x = C4 + C5
        # x = C5
        # x = torch.cat([C3, C4, C5], 1)
        # x = self.fuse([C3, C4, C5])
        # x = x * 100
        x = self.root(x)
        # x = F.relu(x, True)
        return [x, ]


class DenseBlock(BaseModule):
    def __init__(self, in_channels, out_channels, num_layer, kernel, norm_cfg, conv_cfg):
        super(DenseBlock, self).__init__()
        self.conv0 = TorchInitConvModule(in_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg)
        self.conv1 = nn.ModuleList(
            [
                ConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg)
                for _ in range(num_layer)
            ]
        )
        self.conv2 = TorchInitConvModule(out_channels * (num_layer + 1), out_channels, kernel, 1, kernel // 2,
                                         norm_cfg=norm_cfg)

    @auto_fp16()
    # @profile
    def forward(self, x):
        x = self.conv0(x)
        y = [x, ]
        for i in self.conv1:
            x = i(x)
            y.append(x)
        y = torch.cat(y, 1)
        y = self.conv2(y)
        return y


class ResBlock(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg, conv_cfg):
        super(ResBlock, self).__init__()
        kernel = 3
        self.conv = nn.Sequential(
            TorchInitConvModule(in_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, ),
            ConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=None),
        )
        if in_channels != out_channels:
            self.shortcut = TorchInitConvModule(in_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=None)
        else:
            self.shortcut = nn.Sequential()

    @auto_fp16()
    # @profile
    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        x = F.relu(x, True)
        return x


class ResBlock2(BaseModule):
    def __init__(self, in_channels, out_channels, num_layer, kernel, norm_cfg, conv_cfg):
        super(ResBlock2, self).__init__()
        self.conv0 = TorchInitConvModule(in_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg)
        self.conv1 = nn.ModuleList(
            [
                ConvModule(out_channels, out_channels, kernel, 1, kernel // 2, norm_cfg=norm_cfg, conv_cfg=conv_cfg)
                for _ in range(num_layer)
            ]
        )

    @auto_fp16()
    # @profile
    def forward(self, x):
        x = self.conv0(x)
        y = x
        for i, j in enumerate(self.conv1):
            x = j(x)
            # x = j(y / math.sqrt(i + 1))
            y = (y + x)
        y = y / math.sqrt(len(self.conv1) + 1)
        return y


class SA(BaseModule):
    def __init__(self, in_channels):
        super(SA, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 1, 3, 1, 1),
                                  nn.Sigmoid())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.orthogonal_(self.conv[0].weight, gain=1 / math.sqrt(3))
        # init.orthogonal_(self.conv[2].weight, gain=1 / math.sqrt(3))
        init.constant_(self.conv[0].weight, 0)
        init.constant_(self.conv[0].bias, 0)
        # init.constant_(self.conv[2].bias, 0)

    def forward(self, x):
        x = x * self.conv(x) * 2
        return x


class SpaceSelect(BaseModule):
    def __init__(self, in_channels, num):
        super(SpaceSelect, self).__init__()
        # self.conv = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, 1, 3, 1, 1)) for _ in range(num)])
        # self.reset_parameters()

    '''
    def reset_parameters(self) -> None:
        # init.uniform_(self.fc[0].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.uniform_(self.fc[2].weight, -HEAD_INIT_BOUND, HEAD_INIT_BOUND)
        # init.orthogonal_(self.conv[0].weight, gain=1 / math.sqrt(3))
        # init.orthogonal_(self.conv[2].weight, gain=1 / math.sqrt(3))
        for i in self.conv:
            init.constant_(i[0].weight, 0)
            init.constant_(i[0].bias, 0)
        # init.constant_(self.conv[2].bias, 0)
    '''

    def forward(self, x):
        # w = [i(j) for i, j in zip(self.conv, x)]
        outs = torch.stack(x, dim=-1)
        # w = torch.stack(w, dim=-1)
        w = outs
        softmax_outs = F.softmax(w, dim=-1)
        x = (outs * softmax_outs).sum(dim=-1) * 3
        return x


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


class SE(BaseModule):
    def __init__(self, in_channels):
        super(SE, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels),
                                # nn.ReLU(True),
                                # nn.Linear(in_channels, in_channels),
                                nn.Sigmoid())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.fc[0].bias, 0)
        init.constant_(self.fc[0].weight, 0)

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=False)
        x = x * self.fc(y).unsqueeze(2).unsqueeze(2)
        return x


class AddCood(BaseModule):
    def __init__(self):
        super(AddCood, self).__init__()
        self.gamma = nn.Parameter(torch.zeros([1, 2, 1, 1]))
        self.beta = nn.Parameter(torch.ones([1, 2, 1, 1]))

    def forward(self, x):
        b, c, h, w = x.shape
        wh = math.sqrt(h*w)
        gx = torch.arange(w, device=x.device, dtype=x.dtype) / (wh - 1.) * 2
        gy = torch.arange(h, device=x.device, dtype=x.dtype) / (wh - 1.) * 2
        gx = gx.view([1, 1, 1, w]).expand([b, 1, h, w])
        gy = gy.view([1, 1, h, 1]).expand([b, 1, h, w])
        gxgy = torch.cat([gx, gy], 1) * self.beta + self.gamma
        x = torch.cat([x, gxgy], 1)
        return x


class ReLUAdd(BaseModule):
    def __init__(self, n):
        super(ReLUAdd, self).__init__()
        self.weight = nn.Parameter(torch.ones([1, 1, 1, 1, n]) / math.sqrt(n))
        self.bias = nn.Parameter(torch.zeros([1, 1, 1, 1]))

    def forward(self, x):
        x = torch.stack(x, -1)
        x = torch.sum(x * self.weight, dim=-1) + self.bias
        x = F.relu(x, True)
        return x


class WeightAdd(BaseModule):
    def __init__(self, n):
        super(WeightAdd, self).__init__()
        self.weight = nn.Parameter(torch.ones([1, 1, 1, 1, n]))
        # self.bias = nn.Parameter(torch.zeros([1, 1, 1, 1]))

    def forward(self, x):
        x = torch.stack(x, -1)
        x = torch.sum(x * self.weight, dim=-1)# + self.bias
        #x = F.relu(x, True)
        return x

class AddCood2(BaseModule):
    def __init__(self, h, w):
        super(AddCood2, self).__init__()
        self.xy = nn.Parameter(torch.zeros([1, 2, h, w]))
        hw = math.sqrt(h*w)
        gx = torch.arange(w) / (hw - 1.) * 2
        gy = torch.arange(h) / (hw - 1.) * 2
        gx = gx.view([1, 1, 1, w]).expand([1, 1, h, w])
        gy = gy.view([1, 1, h, 1]).expand([1, 1, h, w])
        gxgy = torch.cat([gx, gy], 1)
        self.xy.data = gxgy[:]
        # print(self.xy)

    def forward(self, x):
        x = torch.cat([x, self.xy.expand(x.shape[0], 2, x.shape[2], x.shape[3])], 1)
        return x
