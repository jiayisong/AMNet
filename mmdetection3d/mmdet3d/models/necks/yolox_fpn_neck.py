from mmcv.runner import BaseModule, auto_fp16
import torch.nn as nn
from mmdet.models.builder import NECKS
import torch


def _make_cbl(_in, _out, ks):
    return nn.Sequential(nn.Conv2d(_in, _out, ks, stride=1, padding=ks // 2, bias=False),
                         nn.BatchNorm2d(_out),
                         nn.LeakyReLU(0.1, inplace=True)
                         )


@NECKS.register_module()
class YOLOXFPN(BaseModule):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(self):
        super().__init__()
        self.out0 = self.make_spp_block([512, 1024], 1024)
        # out 1
        self.out1_cbl = _make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = _make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            _make_cbl(in_filters, filters_list[0], 1),
            _make_cbl(filters_list[0], filters_list[1], 3),
            _make_cbl(filters_list[1], filters_list[0], 1),
            _make_cbl(filters_list[0], filters_list[1], 3),
            _make_cbl(filters_list[1], filters_list[0], 1),
        )
        return m

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            _make_cbl(in_filters, filters_list[0], 1),
            _make_cbl(filters_list[0], filters_list[1], 3),
            SPPBottleneck(
                in_channels=filters_list[1],
                out_channels=filters_list[0],
            ),
            _make_cbl(filters_list[0], filters_list[1], 3),
            _make_cbl(filters_list[1], filters_list[0], 1)
        )
        return m

    @auto_fp16()
    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        x2, x1, x0 = inputs
        out_dark5 = self.out0(x0)
        #  yolo branch 1
        x1_in = self.out1_cbl(out_dark5)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, out_dark5)
        return outputs#out_dark3,


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = _make_cbl(in_channels, hidden_channels, 1)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = _make_cbl(conv2_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x
