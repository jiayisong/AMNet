# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from mmcv.utils import deprecated_api_warning
from mmcv.cnn import CONV_LAYERS
from mmcv.utils import ext_loader, print_log

ext_module = ext_loader.load_ext(
    '_ext',
    ['modulated_deform_conv_forward', 'modulated_deform_conv_backward'])


class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding,
                 dilation, groups, deform_groups):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            'mmcv::MMCVModulatedDeformConv2d',
            *input_tensors,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            groups_i=groups,
            deform_groups_i=deform_groups)

    @staticmethod
    def forward(ctx,
                input,
                offset,
                mask,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deform_groups=1):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        # offset = offset * 10
        input = input.type_as(offset)
        weight = weight.type_as(input)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConv2dFunction._output_size(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_output = grad_output.contiguous()
        ext_module.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        # grad_offset = grad_offset * 10
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


class ModulatedDeformConv2d(nn.Module):

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='ModulatedDeformConv2d')
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ModulatedDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@CONV_LAYERS.register_module('myDCNv2')
class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
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

    _version = 2

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ModulatedDeformConv2dPack, self).__init__(in_channels,
                                                        # out_channels - deform_groups * 3 * kernel_size * kernel_size,
                                                        out_channels,
                                                        kernel_size,
                                                        stride=stride,
                                                        padding=padding,
                                                        dilation=dilation,
                                                        groups=groups,
                                                        deform_groups=deform_groups,
                                                        bias=bias)
        '''
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            #nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation,
                      # bias=False
                      )
        )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            #n.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, self.deform_groups * self.kernel_size[0] * self.kernel_size[1],
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation,
                      # bias=False
                      ),
            nn.Sigmoid()
        )
        '''
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.in_channels, self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation,
                      # bias=False
                      )
        )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(self.in_channels, self.deform_groups * self.kernel_size[0] * self.kernel_size[1],
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation,
                      # bias=False
                      ),
            nn.Sigmoid()
        )
        #'''
        '''
        self.fc_mask = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.in_channels, self.out_channels),

            nn.Sigmoid(),
            nn.Unflatten(1, (self.out_channels, 1, 1))
        )
        '''
        # self.bn = nn.BatchNorm2d(deform_groups * 3 * kernel_size * kernel_size)
        # self.bn = nn.BatchNorm2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1])
        # self.bn2 = nn.BatchNorm2d(self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1])
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConv2dPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()
            # self.conv_offset.reset_parameters()
            # self.bn.bias.data.zero_()
            # self.bn.weight.data.zero_()
        if hasattr(self, 'conv_mask'):
            self.conv_mask[-2].weight.data.zero_()
            self.conv_mask[-2].bias.data.zero_()
        if hasattr(self, 'fc_mask'):
            self.fc_mask[-3].weight.data.zero_()
            self.fc_mask[-3].bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.conv_mask(x)
        # out = self.bn(out)
        # x_debug = x.detach().cpu().numpy()
        # a = 10 / 2
        # b, c, h, w = x.shape
        # o1, o2, mask = torch.chunk(out, 3, dim=1)
        # o1 = w * (4 * torch.sigmoid(o1 / w) - 2)
        # o2 = h * (4 * torch.sigmoid(o2 / h) - 2)
        # offset = torch.cat((o1, o2), dim=1)

        # offset = a * (4 * torch.sigmoid(offset / a) - 2)
        # o1_debug = o1.detach().cpu().numpy()
        # o2_debug = o2.detach().cpu().numpy()
        # print(self.weight.abs().max(), self.weight.abs().min())
        # print(o1.abs().mean())
        # offset = self.bn2(offset)
        # mask = self.bn1(mask)
        # mask = torch.sigmoid(mask)
        # print(offset.shape)
        dcn = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                      self.stride, self.padding,
                                      self.dilation, self.groups,
                                      self.deform_groups)
        # dcn = dcn * self.fc_mask(x)
        # out = self.bn(out.detach())
        # dcn = torch.cat((dcn, out), 1)
        return dcn

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
