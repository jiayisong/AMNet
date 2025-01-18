# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
from multiprocessing import Pool
from mmcv.utils import deprecated_api_warning
from ..cnn import CONV_LAYERS
from ..utils import ext_loader, print_log

ext_module = ext_loader.load_ext(
    '_ext',
    ['modulated_deform_conv_fast_forward', 'modulated_deform_conv_fast_backward'])


def backward_kernel(grad_output, offset, input, offset_temp, deform_groups, kernel_h, kernel_w, height, width,
                    channels_per_group, b, g, i, j):
    columns = grad_output[((b * deform_groups + g) * kernel_h + i) * kernel_w + j]
    wh_mat = columns.new_zeros([width * height, height * width])
    ext_module.modulated_deform_conv_fast_backward(columns, offset[b][g][i][j], input[b][g], wh_mat,
                                                   offset_temp,
                                                   channels_per_group=channels_per_group,
                                                   height=height, width=width)
    offset_sum = torch.sum(offset_temp, dim=1, keepdim=False)
    temp = torch.mm(columns.view(-1, height * width), wh_mat)
    return offset_sum, temp


class ModulatedDeformConv2dFastFunction(Function):

    @staticmethod
    #@profile
    def forward(ctx, input, offset, kernel, deform_groups=1):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.kernel = _pair(kernel)
        ctx.deform_groups = deform_groups
        b, c, h, w = input.shape
        B, C, H, W = offset.shape
        channels_per_group = c // deform_groups
        assert b == B and H == h and W == w and C == ctx.kernel[0] * ctx.kernel[
            1] * deform_groups * 2 and c % deform_groups == 0 and deform_groups <= c

        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        ctx.save_for_backward(input, offset)
        output = input.new_zeros([b, ctx.kernel[0] * ctx.kernel[1] * c, h, w])
        ext_module.modulated_deform_conv_fast_forward(offset, input, output, kernel_h=ctx.kernel[0],
                                                      kernel_w=ctx.kernel[1],
                                                      height=h, width=w, batch_size=b, deform_groups=deform_groups,
                                                      channels_per_group=channels_per_group)
        return output

    @staticmethod
    @once_differentiable
    #@profile
    def backward(ctx, grad_output):

        input, offset = ctx.saved_tensors
        kernel_h = ctx.kernel[0]
        kernel_w = ctx.kernel[1]
        deform_groups = ctx.deform_groups

        grad_offset = []  # torch.zeros_like(offset)

        grad_output = grad_output.contiguous()
        batch, channels, height, width = input.shape
        _, channels_out, height_out, width_out = grad_output.shape
        channels_per_group = channels // deform_groups
        grad_input = []  # grad_input.view(batch, channels * height * width)
        # return torch.zeros_like(input), torch.zeros_like(offset), None, None, None
        # columns = grad_output.new_zeros([channels * kernel_h * kernel_w, height_out * width_out])
        # add_index = grad_output.new_zeros([height * width * height_out * width_out // 2])
        # wh_mat = grad_output.new_zeros([height_out * width_out, height * width])
        offset_temp = grad_output.new_zeros([2, channels_per_group, height, width])
        grad_output = grad_output.view(batch * channels_out, height_out, width_out)
        grad_output = torch.split(grad_output, channels_per_group, 0)
        offset = offset.view(batch, deform_groups, kernel_h, kernel_w, 2, height, width).contiguous()
        input = input.view(batch, deform_groups, channels_per_group, height, width).contiguous()
        for b in range(batch):
            for g in range(deform_groups):
                ij_sum = 0
                '''
                process_args = []
                for i in range(kernel_h):
                    for j in range(kernel_w):
                        process_args.append((grad_output, offset, input, offset_temp, deform_groups, kernel_h, kernel_w,
                                        height, width,
                                        channels_per_group, b, g, i, j))

                with Pool(kernel_h*kernel_w) as p:
                    outputs = p.map(backward_kernel, process_args)
                for i in range(kernel_h):
                    for j in range(kernel_w):
                        offset_sum, temp = outputs[i*kernel_w+j]
                        ij_sum = ij_sum + temp
                        grad_offset.append(offset_sum)
                '''
                for i in range(kernel_h):
                    for j in range(kernel_w):
                        offset_sum, temp = backward_kernel(grad_output, offset, input, offset_temp, deform_groups, kernel_h, kernel_w,
                                        height, width,
                                        channels_per_group, b, g, i, j)
                        ij_sum = ij_sum + temp
                        grad_offset.append(offset_sum)

                grad_input.append(ij_sum)
        grad_input = torch.stack(grad_input, 0).view(batch, channels, height, width)
        grad_offset = torch.stack(grad_offset, 0).view(batch, -1, height, width)
        return grad_input, grad_offset, None, None, None


modulated_deform_conv2d_fast = ModulatedDeformConv2dFastFunction.apply


class ModulatedDeformConv2dFast(nn.Module):

    def __init__(self,
                 kernel_size,
                 deform_groups=1):
        super(ModulatedDeformConv2dFast, self).__init__()
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups

    def forward(self, x, offset):
        return modulated_deform_conv2d_fast(x, offset, self.kernel_size, self.deform_groups)


@CONV_LAYERS.register_module('DCNv2Fast')
class ModulatedDeformConv2dFastPack(nn.Module):
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

    def __init__(self, in_channels, out_channels, kernel_size, dilation, deform_groups=1, groups=1, bias=False,
                 **kwargs):
        super(ModulatedDeformConv2dFastPack, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.deform_groups = deform_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 1
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.dcn = ModulatedDeformConv2dFast(kernel_size, deform_groups)
        self.conv_offset = nn.Conv2d(in_channels,
                                     self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                                     kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=self.dilation, bias=True)
        self.conv = nn.Conv2d(in_channels * self.kernel_size[0] * self.kernel_size[1],
                              out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)
        self.init_weights()

    def init_weights(self):
        self.conv.reset_parameters()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask).view(b, self.deform_groups, self.kernel_size[0], self.kernel_size[1], 1, h, w)
        y = self.dcn(x, offset).view(b, self.deform_groups, self.kernel_size[0], self.kernel_size[1], -1, h, w)
        y = self.conv((y * mask).view(b, -1, h, w))
        return y

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'ModulatedDeformConvPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
