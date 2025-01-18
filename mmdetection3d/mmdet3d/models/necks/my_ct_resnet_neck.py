import torch
from mmcv.cnn import xavier_init
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


@NECKS.register_module()
class MyCTResNetNeck(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channel,
                 out_channel):
        super(MyCTResNetNeck, self).__init__(None)

        self.fp16_enabled = False

        self.in_channel = in_channel
        K = (5,9,13)
        a = len(K) + 1
        inner_channel=in_channel[-1] * a
        self.conv = nn.Sequential(
            #SPP(K),
            #nn.Conv2d(inner_channel, inner_channel, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            D2S(inner_channel, 4),
            nn.Conv2d(inner_channel // 16, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    @auto_fp16()
    def forward(self, x):
        C3, C4, C5 = x
        x = self.conv(C5)
        return [x]


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


class SPP(nn.Module):
    def __init__(self, k=(3,5,7,9,11)):
        super(SPP, self).__init__()
        self.pool = nn.ModuleList()
        for i in k:
            self.pool.append(nn.MaxPool2d(i, 1, i // 2))

    def forward(self, x):
        y = [x]
        for i in self.pool:
            y.append(i(x))
        return torch.cat(y, dim=1)