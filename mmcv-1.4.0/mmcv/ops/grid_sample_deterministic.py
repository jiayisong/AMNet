# Copyright (c) OpenMMLab. All rights reserved.
import math
from mmcv.cnn.utils import uniform_init, normal_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
from multiprocessing import Pool
from mmcv.utils import deprecated_api_warning
from ..cnn import CONV_LAYERS
from ..utils import ext_loader, print_log

ext_module = ext_loader.load_ext(
    '_ext',
    ['grid_sample_deterministic_forward', 'grid_sample_deterministic_input_backward',
     'grid_sample_deterministic_offset_backward'])


class GridSampleDeterministicFunction(Function):

    @staticmethod
    # @profile
    def forward(ctx, input, offset):
        b, c, h, w = input.shape
        B, C, N = offset.shape
        assert b == B and C == 2 and input.dim() == 4 and offset.dim() == 3

        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        ctx.save_for_backward(input, offset)
        output = input.new_zeros([b, c, N])
        ext_module.grid_sample_deterministic_forward(input, offset, output, N, c, h, w)
        return output

    @staticmethod
    @once_differentiable
    # @profile
    def backward(ctx, grad_output):
        input, offset = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        input = input.contiguous()
        offset = offset.contiguous()
        # offset = torch.rand_like(offset) * 100
        b, c, h, w = input.shape
        _, _, N = offset.shape
        grad_input = offset.new_zeros([b, c, h, w])
        grad_offset = offset.new_zeros([b, 2, N])
        t1 = torch.floor(offset)

        t2 = t1[:, 0, :].clamp(0, w - 1) + t1[:, 1, :].clamp(0, h - 1) * w
        t2_sort, t2_indice = t2.sort(dim=1)
        t2_change = t2_sort[:, 1:] - t2_sort[:, :-1]
        t2_mask = torch.cat((t2_change.new_ones([b, 1], dtype=torch.bool), t2_change > 0), 1)
        batch_id, t2_i_i_s_id = torch.where(t2_mask)
        t2_i_i_f_id = torch.cat((t2_i_i_s_id[1:], t2_i_i_s_id.new_zeros([1, ])), 0)
        loc_number = t2_sort[t2_mask]
        w_number = loc_number % w
        h_number = torch.div(loc_number, w, rounding_mode="trunc")
        w_ji = (w_number % 2).bool()
        w_ou = ~w_ji
        h_ji = (h_number % 2).bool()
        h_ou = ~h_ji
        g1 = w_ou * h_ou
        # '''
        if g1.any():
            batch_id1, t2_i_i_s_id1, t2_i_i_f_id1 = batch_id[g1], t2_i_i_s_id[g1], t2_i_i_f_id[g1]
            # print(t2_i_i_f_id1.shape[0])
            # print(batch_id1, t2_i_i_s_id1, t2_i_i_f_id1)
            ext_module.grid_sample_deterministic_input_backward(grad_output, t2_indice, offset, batch_id1, t2_i_i_s_id1,
                                                                t2_i_i_f_id1, grad_input, t2_i_i_f_id1.shape[0], N, c,
                                                                h, w)
        # print(1, grad_input)
        g1 = w_ou * h_ji
        if g1.any():
            batch_id1, t2_i_i_s_id1, t2_i_i_f_id1 = batch_id[g1], t2_i_i_s_id[g1], t2_i_i_f_id[g1]
            # print(batch_id1, t2_i_i_s_id1, t2_i_i_f_id1)
            ext_module.grid_sample_deterministic_input_backward(grad_output, t2_indice, offset, batch_id1, t2_i_i_s_id1,
                                                                t2_i_i_f_id1, grad_input, t2_i_i_f_id1.shape[0], N, c,
                                                                h, w)
        # print(2, grad_input)
        g1 = w_ji * h_ou
        if g1.any():
            batch_id1, t2_i_i_s_id1, t2_i_i_f_id1 = batch_id[g1], t2_i_i_s_id[g1], t2_i_i_f_id[g1]
            # print(t2_i_i_f_id1.shape[0])
            # print(batch_id1, t2_i_i_s_id1, t2_i_i_f_id1)
            ext_module.grid_sample_deterministic_input_backward(grad_output, t2_indice, offset, batch_id1, t2_i_i_s_id1,
                                                                t2_i_i_f_id1, grad_input, t2_i_i_f_id1.shape[0], N, c,
                                                                h, w)
        # print(3, grad_input)
        g1 = w_ji * h_ji
        if g1.any():
            batch_id1, t2_i_i_s_id1, t2_i_i_f_id1 = batch_id[g1], t2_i_i_s_id[g1], t2_i_i_f_id[g1]
            # print(t2_i_i_f_id1.shape[0])
            # print(batch_id1, t2_i_i_s_id1, t2_i_i_f_id1)
            ext_module.grid_sample_deterministic_input_backward(grad_output, t2_indice, offset, batch_id1, t2_i_i_s_id1,
                                                                t2_i_i_f_id1, grad_input, t2_i_i_f_id1.shape[0], N, c,
                                                                h, w)
        # print(4, grad_input)
        # del g1, batch_id1, t2_i_i_s_id1, t2_i_i_f_id1, t2_indice
        # '''
        ext_module.grid_sample_deterministic_offset_backward(grad_output, input, offset, grad_offset, N, c, h, w)
        # grad_offset[:] = 0.0
        # grad_input[:] = 0.0001
        # print(grad_offset.mean())
        # print(grad_input)
        return grad_input, grad_offset


grid_sample_deterministic = GridSampleDeterministicFunction.apply


@CONV_LAYERS.register_module('DCNv2Fastv2')
class ModulatedDeformConv2dFastv2Pack(nn.Module):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 **kwargs):
        super(ModulatedDeformConv2dFastv2Pack, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.mask = kwargs.pop('mask', True)

        # self.offset_factor = kwargs['offset_factor']
        a = 3 if self.mask else 2
        self.offset_factor = 1#math.sqrt(128 / in_channels)
        self.conv_offset = nn.Conv2d(in_channels, a * self.kernel_size[0] * self.kernel_size[1],
                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, bias=True)
        self.x_start = self.dilation[1] / 2 * (self.kernel_size[1] - 1) - self.padding[1]
        self.y_start = self.dilation[0] / 2 * (self.kernel_size[0] - 1) - self.padding[0]
        #self.conv0 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = nn.Conv2d(in_channels * self.kernel_size[0] * self.kernel_size[1],
                              out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)

        self.init_weights()

    def init_weights(self):
        # n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        # stdv = 1. / math.sqrt(n*3)
        # uniform_init(self.conv, -stdv, stdv)
        # normal_init(self.conv, 0, stdv)
        # uniform_init(self.conv, -0.001, 0.001)
        #torch.nn.init.orthogonal_(self.conv0.weight, gain=1 / math.sqrt(3))
        torch.nn.init.orthogonal_(self.conv.weight, gain=1 / math.sqrt(3))
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()
            x_start = self.dilation[1] * (1 - self.kernel_size[1]) / 2
            x_cood = torch.arange(x_start, x_start + self.dilation[1] * self.kernel_size[1], self.dilation[1])
            y_start = self.dilation[0] * (1 - self.kernel_size[0]) / 2
            y_cood = torch.arange(y_start, y_start + self.dilation[0] * self.kernel_size[0], self.dilation[0])
            x_cood, y_cood = torch.meshgrid(x_cood, y_cood, indexing='xy')
            x_bias_bias = x_cood.reshape(1, -1, 1, 1)
            y_bias_bias = y_cood.reshape(1, -1, 1, 1)
            self.register_buffer('x_bias_bias', x_bias_bias)
            self.register_buffer('y_bias_bias', y_bias_bias)
            # print(self.conv_offset.bias.data)

    def forward(self, x):

        out = self.conv_offset(x)
        #x = self.conv0(x)
        b, c, h, w = x.shape
        _, _, H, W = out.shape
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            mask = torch.sigmoid(mask).view(b, 1, self.kernel_size[0], self.kernel_size[1], H, W)
        else:
            o1, o2 = torch.chunk(out, 2, dim=1)
        # offset = torch.cat((o1, o2), dim=1).view(b, 1,1, 2, H, W)
        # o1 = offset[:,:,:,1,:,:].view(b,1,H,W)
        # o2 = offset[:,:,:,0,:,:].view(b,1,H,W)
        x_cood = torch.arange(self.x_start, self.x_start + self.stride[1] * W, self.stride[1], device=o1.device)
        o1 = self.offset_clamp(o1) + x_cood.view(1, 1, 1, -1) + self.x_bias_bias
        # o1 = torch.clamp(o1, -self.padding[1], w - 1 + self.padding[1])
        y_cood = torch.arange(self.y_start, self.y_start + self.stride[0] * H, self.stride[0], device=o2.device)
        o2 = self.offset_clamp(o2) + y_cood.view(1, 1, -1, 1) + self.y_bias_bias
        # o2 = torch.clamp(o2, -self.padding[0], h - 1 + self.padding[0])
        offset = torch.stack((o1, o2), dim=1)
        # print(offset[0,:,:,10,5])
        offset = offset.view(b, 2, -1)
        y = grid_sample_deterministic(x, offset).view(b, c, self.kernel_size[0], self.kernel_size[1], H, W)
        if self.mask:
            y = self.conv((y * mask).view(b, -1, H, W))
        else:
            y = self.conv(y.view(b, -1, H, W))
        return y

    def offset_clamp(self, x):
        return x * self.offset_factor
        # return F.tanh(x / 64) * self.offset_factor


class DeformUpSample(nn.Module):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, in_channels, scale_factor=2, kernel_size=1, mask=False):
        super(DeformUpSample, self).__init__()
        a = 3 if mask else 2
        self.conv_offset = nn.Conv2d(in_channels, a * scale_factor ** 2, kernel_size=kernel_size, stride=1,
                                     padding=kernel_size // 2, dilation=1, bias=True)
        self.scale_factor = scale_factor
        self.offset_factor = math.sqrt(128 / (in_channels))
        self.mask = mask
        self.init_weights()

    def init_weights(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()
            x_cood = torch.arange(0, 1, 1 / self.scale_factor)
            y_cood = torch.arange(0, 1, 1 / self.scale_factor)
            x_cood, y_cood = torch.meshgrid(x_cood, y_cood, indexing='xy')
            self.register_buffer('x_bias_bias', x_cood.reshape(1, -1, 1, 1))
            self.register_buffer('y_bias_bias', y_cood.reshape(1, -1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.conv_offset(x)
        _, _, H, W = out.shape
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
        else:
            o1, o2 = torch.chunk(out, 2, dim=1)

        # offset = torch.cat((o1, o2), dim=1).view(b, 1,1, 2, H, W)
        # o1 = offset[:,:,:,1,:,:].view(b,1,H,W)
        # o2 = offset[:,:,:,0,:,:].view(b,1,H,W)
        x_cood = torch.arange(0, W, 1, device=o1.device)
        o1 = self.offset_factor * o1 + x_cood.view(1, 1, 1, -1) + self.x_bias_bias
        y_cood = torch.arange(0, H, 1, device=o2.device)
        o2 = self.offset_factor * o2 + y_cood.view(1, 1, -1, 1) + self.y_bias_bias
        offset = torch.stack((o1, o2), dim=1)
        # print(offset[0,:,:,10,5])
        # print(offset)
        offset = offset.view(b, 2, self.scale_factor, self.scale_factor, H, W).permute(0, 1, 4, 2, 5, 3).reshape(b, 2,
                                                                                                                 -1)
        y = grid_sample_deterministic(x, offset).view(b, c, self.scale_factor * H, self.scale_factor * W)
        if self.mask:
            mask = torch.sigmoid(mask).view(b, 1, self.scale_factor, self.scale_factor, H, W).permute(0, 1, 4, 2, 5,
                                                                                                      3).reshape(b, 1,
                                                                                                                 self.scale_factor * H,
                                                                                                                 self.scale_factor * W)
            y = y * mask
        return y

@CONV_LAYERS.register_module('DCNTrans')
class DeformTransConv(nn.Module):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 **kwargs):
        super(DeformTransConv, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.mask = kwargs.pop('mask', True)

        # self.offset_factor = kwargs['offset_factor']
        a = 3 if self.mask else 2
        self.offset_factor = 1#math.sqrt(128 / in_channels)
        self.conv_offset = nn.Conv2d(in_channels,
                                     a * self.kernel_size[0] * self.kernel_size[1] * self.stride[0] * self.stride[1],
                                     kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                                     dilation=dilation, bias=True)
        self.x_start = self.dilation[1] / 2 * (self.kernel_size[1] - 1) - self.padding[1]
        self.y_start = self.dilation[0] / 2 * (self.kernel_size[0] - 1) - self.padding[0]
        #self.conv0 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = nn.Conv2d(in_channels * self.kernel_size[0] * self.kernel_size[1],
                              out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)

        self.init_weights()

    def init_weights(self):
        # n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        # stdv = 1. / math.sqrt(n*3)
        # uniform_init(self.conv, -stdv, stdv)
        # normal_init(self.conv, 0, stdv)
        # uniform_init(self.conv, -0.001, 0.001)
        #torch.nn.init.orthogonal_(self.conv0.weight, gain=1 / math.sqrt(3))
        torch.nn.init.orthogonal_(self.conv.weight, gain=1 / math.sqrt(3))
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()
            x_start = self.dilation[1] * (1 - self.kernel_size[1]) / 2
            x_cood = torch.arange(x_start, x_start + self.dilation[1] * self.kernel_size[1], self.dilation[1])
            y_start = self.dilation[0] * (1 - self.kernel_size[0]) / 2
            y_cood = torch.arange(y_start, y_start + self.dilation[0] * self.kernel_size[0], self.dilation[0])
            x_cood, y_cood = torch.meshgrid(x_cood, y_cood, indexing='xy')
            x_bias_bias = x_cood.reshape(1, -1, 1, 1, 1)
            y_bias_bias = y_cood.reshape(1, -1, 1, 1, 1)
            x_stride_bias = torch.arange(0, 1, 1 / self.stride[1])
            y_stride_bias = torch.arange(0, 1, 1 / self.stride[0])
            x_stride, y_stride = torch.meshgrid(x_stride_bias, y_stride_bias, indexing='xy')
            x_stride_bias_bias = x_stride.reshape(1, 1, -1, 1, 1)
            y_stride_bias_bias = y_stride.reshape(1, 1, -1, 1, 1)
            self.register_buffer('x_bias_bias', x_bias_bias)
            self.register_buffer('y_bias_bias', y_bias_bias)
            self.register_buffer('x_stride_bias_bias', x_stride_bias_bias)
            self.register_buffer('y_stride_bias_bias', y_stride_bias_bias)
            # print(self.conv_offset.bias.data)

    def forward(self, x):

        out = self.conv_offset(x)
        #x = self.conv0(x)
        b, c, h, w = x.shape
        _, _, H, W = out.shape
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            mask = torch.sigmoid(mask).view(b, 1, self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], H, W)
        else:
            o1, o2 = torch.chunk(out, 2, dim=1)
        # offset = torch.cat((o1, o2), dim=1).view(b, 1,1, 2, H, W)
        # o1 = offset[:,:,:,1,:,:].view(b,1,H,W)
        # o2 = offset[:,:,:,0,:,:].view(b,1,H,W)
        x_cood = torch.arange(self.x_start, self.x_start + W, 1, device=o1.device)
        o1 = self.offset_clamp(o1).view(b, self.kernel_size[0] * self.kernel_size[1], self.stride[0] * self.stride[1], H, W)\
             + x_cood.view(1, 1, 1, 1, -1) + self.x_bias_bias + self.x_stride_bias_bias
        # o1 = torch.clamp(o1, -self.padding[1], w - 1 + self.padding[1])
        y_cood = torch.arange(self.y_start, self.y_start + H, 1, device=o2.device)
        o2 = self.offset_clamp(o2).view(b, self.kernel_size[0] * self.kernel_size[1], self.stride[0] * self.stride[1], H, W)\
             + y_cood.view(1, 1, 1, -1, 1) + self.y_bias_bias + self.y_stride_bias_bias
        # o2 = torch.clamp(o2, -self.padding[0], h - 1 + self.padding[0])
        offset = torch.stack((o1, o2), dim=1)
        #print(offset)
        # print(offset[0,:,:,10,5])
        offset = offset.view(b, 2, -1)

        y = grid_sample_deterministic(x, offset).view(b, c, self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], H, W)
        if self.mask:
            y = y * mask
        y = y.permute(0, 1, 2, 3, 6, 4, 7, 5).reshape(b, -1, self.stride[0]*H, W*self.stride[1])
        y = self.conv(y)
        return y

    def offset_clamp(self, x):
        return x * self.offset_factor
        # return F.tanh(x / 64) * self.offset_factor
