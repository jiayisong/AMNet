# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from mmcv.utils import deprecated_api_warning
from ..cnn import CONV_LAYERS
from ..utils import ext_loader, print_log

ext_module = ext_loader.load_ext(
    '_ext',
    ['modulated_deform_conv_forward', 'modulated_deform_conv_deterministic_forward',
     'modulated_deform_conv_deterministic_backward', 'modulated_deform_conv_backward',
     'modulated_deformable_col2im_deterministic'])


def forward2(ctx,input,offset,mask,weight,bias=None,stride=1,padding=0,
             dilation=1,groups=1,deform_groups=1):
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
    input = input.type_as(offset)
    weight = weight.type_as(input)
    bias = bias.type_as(input)
    # mask = torch.ones_like(mask)
    # offset = torch.zeros_like(offset)
    # ctx.save_for_backward(input, offset, mask, weight, bias)
    output = input.new_empty(
        ModulatedDeformConv2dDeterministicFunction._output_size(ctx, input, weight))
    # ctx._bufs = [input.new_empty(0), input.new_empty(0)]
    ext_module.modulated_deform_conv_forward(
        input,
        weight,
        bias,
        input.new_empty(0),
        offset,
        mask,
        output,
        input.new_empty(0),
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
        with_bias=ctx.with_bias,
    )
    return output


def backward2(ctx, grad_output):
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
        grad_weight.new_empty(0),
        offset,
        mask,
        grad_weight.new_empty(0),
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

    return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
            None, None, None, None, None)


class ModulatedDeformConv2dDeterministicFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding,
                 dilation, groups, deform_groups):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            'mmcv::MMCVModulatedDeformConv2dDeterministic',
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
        #output2 = forward2(ctx, input, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups)
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
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)
        # mask = torch.ones_like(mask)
        # offset = torch.zeros_like(offset)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConv2dDeterministicFunction._output_size(ctx, input, weight))
        # ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            input.new_empty(0),
            offset,
            mask,
            output,
            input.new_empty(0),
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
            with_bias=ctx.with_bias,
            #im2col_step=2,
        )
        #print(input.shape)

        #print((output2 - output).abs().max(), (output2).abs().mean(), (output2 - output).abs().mean(), )

        return output

    @staticmethod
    @once_differentiable
    #@profile
    def backward(ctx, grad_output):
        # grad_output = grad_output * 1000
        input, offset, mask, weight, bias = ctx.saved_tensors
        # grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        grad_output = grad_output.contiguous()
        batch, channels, height, width = input.shape
        _, channels_out, height_out, width_out = grad_output.shape
        kernel_h = weight.size(2)
        kernel_w = weight.size(3)
        group = ctx.groups

        grad_input = []  # grad_input.view(batch, channels * height * width)
        columns = grad_output.new_zeros([channels * kernel_h * kernel_w, height_out * width_out])
        # add_index = grad_output.new_zeros([height * width * height_out * width_out // 2])
        # add_value = grad_output.new_zeros([height_out * width_out * height * width])
        grad_output = grad_output.view(grad_output.size(0), group, grad_output.size(1) // group, grad_output.size(2),
                                       grad_output.size(3))
        # add_value2 = grad_output.new_zeros([height_out * width_out * height * width * 8])
        for b in range(batch):
            columns = columns.view([group, columns.size(0) // group, columns.size(1)])
            weight = weight.view([group, weight.size(0) // group, weight.size(1), weight.size(2), weight.size(3)])
            for g in range(group):
                columns[g].addmm_(weight[g].flatten(1).transpose(0, 1), grad_output[b][g].flatten(1), beta=0, alpha=1)

            columns = columns.view([group, channels // group, kernel_h, kernel_w, height_out * width_out])
            gg = []
            for g in range(group):
                ij_sum = 0
                for i in range(kernel_h):
                    for j in range(kernel_w):
                        columns_gij = columns[g, :, i, j, :]
                        mask_gij = mask.view(batch, group, 1, kernel_h, kernel_w, height_out * width_out)[b, g, :, i, j,
                                   :]
                        columns_gij_mask = columns_gij * mask_gij
                        # print(columns_gij)
                        # add_value.data.fill_(0)
                        add_value = grad_output.new_zeros([height_out * width_out, height * width])
                        # print(height_out, width_out, height, width, ctx.stride, ctx.padding, ctx.dilation, i, j, g, offset[b].shape)
                        ext_module.modulated_deformable_col2im_deterministic(offset[b], add_value, add_value,
                                                                             height=height, width=width,
                                                                             height_out=height_out, width_out=width_out,
                                                                             kh_id=i, kw_id=j,
                                                                             kernel_h=kernel_h,
                                                                             kernel_w=kernel_w,
                                                                             stride_h=ctx.stride[0],
                                                                             stride_w=ctx.stride[1],
                                                                             pad_h=ctx.padding[0], pad_w=ctx.padding[1],
                                                                             dilation_h=ctx.dilation[0],
                                                                             dilation_w=ctx.dilation[1],
                                                                             g_id=g)
                        camm = torch.mm(columns_gij_mask, add_value)
                        # add_value_debug = add_value.cpu().numpy()
                        # print(add_value.view(height_out * width_out, height * width))
                        # print(camm)
                        ij_sum = ij_sum + camm
                gg.append(ij_sum)
            gi = torch.cat(gg, dim=0)
            # add_index2 = add_index.int()[:kernel_h * kernel_w * height_out * width_out * 4]
            # add_value = add_value[:kernel_h * kernel_w * height_out * width_out * 4]
            # gi = torch.sparse_coo_tensor(add_index2.unsqueeze(0), add_value, (channels*height*width,))
            # gi = gi.to_dense()
            # zero = grad_output.new_zeros([channels*height*width])
            # gi = torch.index_add(zero, 0, add_index2, add_value)
            grad_input.append(gi)
            columns = columns.view(channels * kernel_h * kernel_w, height_out * width_out)
            weight = weight.view([weight.size(0) * weight.size(1), weight.size(2), weight.size(3), weight.size(4)])
            ext_module.modulated_deform_conv_deterministic_backward(input[b], offset[b], mask[b],
                                                                    columns, add_value, add_value,
                                                                    grad_offset[b], grad_mask[b],
                                                                    kernel_h=kernel_h,
                                                                    kernel_w=kernel_w,
                                                                    stride_h=ctx.stride[0], stride_w=ctx.stride[1],
                                                                    pad_h=ctx.padding[0], pad_w=ctx.padding[1],
                                                                    dilation_h=ctx.dilation[0],
                                                                    dilation_w=ctx.dilation[1],
                                                                    deformable_group=ctx.deform_groups)

            columns = columns.view([group, columns.size(0) // group, columns.size(1)])
            grad_weight = grad_weight.view(
                [group, grad_weight.size(0) // group, grad_weight.size(1), grad_weight.size(2),
                 grad_weight.size(3)])
            if ctx.with_bias:
                grad_bias = grad_bias.view(group, grad_bias.size(0) // group)

            for g in range(group):
                grad_weight[g] = grad_weight[g].flatten(1).addmm_(grad_output[b][g].flatten(1),
                                                                  columns[g].transpose(0, 1)).view_as(grad_weight[g])
                if ctx.with_bias:
                    grad_bias[g] = torch.sum(grad_bias[g].add_(grad_output[b][g].flatten(1)), 1, keep_dim=False)

            grad_weight = grad_weight.view(grad_weight.size(0) * grad_weight.size(1), grad_weight.size(2),
                                           grad_weight.size(3),
                                           grad_weight.size(4))
            if ctx.with_bias:
                grad_bias = grad_bias.view(grad_bias.size(0) * grad_bias.size(1))

            columns = columns.view(channels * kernel_h * kernel_w, height_out * width_out)

        grad_output = grad_output.view(grad_output.size(0) * grad_output.size(1), grad_output.size(2),
                                       grad_output.size(3),
                                       grad_output.size(4))
        grad_input = torch.stack(grad_input, 0).view(batch, channels, height, width)
        if not ctx.with_bias:
            grad_bias = None
        # grad_input2, grad_offset2, grad_mask2, grad_weight2, grad_bias2, _, _, _, _, _ = backward2(ctx, grad_output)
        # grad_input3, grad_offset2, grad_mask2, grad_weight2, grad_bias2, _, _, _, _, _ = backward2(ctx, grad_output)
        # print((grad_input2 - grad_input).abs().max(), (grad_input2).abs().mean(), (grad_input2 - grad_input).abs().mean(), )
        print(grad_input)
        # print(grad_input2)

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None)

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


modulated_deform_conv2d_deterministic = ModulatedDeformConv2dDeterministicFunction.apply


class ModulatedDeformConv2dDeterministic(nn.Module):

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
        super(ModulatedDeformConv2dDeterministic, self).__init__()
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
        return modulated_deform_conv2d_deterministic(x, offset, mask, self.weight, self.bias,
                                                     self.stride, self.padding,
                                                     self.dilation, self.groups,
                                                     self.deform_groups)


@CONV_LAYERS.register_module('DCNv2Deterministic')
class ModulatedDeformConv2dDeterministicPack(ModulatedDeformConv2dDeterministic):
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

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dDeterministicPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        #self.conv_se = nn.Linear(self.in_channels, self.in_channels)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConv2dDeterministicPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()
        if hasattr(self, 'conv_se'):
            self.conv_se.weight.data.zero_()
            self.conv_se.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)

        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # se = self.conv_se(torch.mean(x, dim=[2,3], keepdim=False)).unsqueeze(2).unsqueeze(2)
        #x = x * torch.sigmoid(se) * 2
        return modulated_deform_conv2d_deterministic(x, offset, mask, self.weight, self.bias,
                                                     self.stride, self.padding,
                                                     self.dilation, self.groups,
                                                     self.deform_groups)

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
