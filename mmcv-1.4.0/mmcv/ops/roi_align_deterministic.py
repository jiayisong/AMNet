# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import deprecated_api_warning, ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['roi_align_forward', 'roi_align_backward', 'roi_align_deterministic_backward'])


def backward2(ctx, grad_output):
    rois, argmax_y, argmax_x = ctx.saved_tensors
    grad_input = grad_output.new_zeros(ctx.input_shape)
    # complex head architecture may cause grad_output uncontiguous.
    grad_output = grad_output.contiguous()
    ext_module.roi_align_backward(
        grad_output,
        rois,
        argmax_y,
        argmax_x,
        grad_input,
        aligned_height=ctx.output_size[0],
        aligned_width=ctx.output_size[1],
        spatial_scale=ctx.spatial_scale,
        sampling_ratio=ctx.sampling_ratio,
        pool_mode=ctx.pool_mode,
        aligned=ctx.aligned)
    return grad_input, None, None, None, None, None, None


class RoIAlignDeterministicFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
                 pool_mode, aligned):
        from ..onnx import is_custom_op_loaded
        has_custom_op = is_custom_op_loaded()
        if has_custom_op:
            return g.op(
                'mmcv::MMCVRoiAlignDeterministic',
                input,
                rois,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=sampling_ratio,
                mode_s=pool_mode,
                aligned_i=aligned)
        else:
            from torch.onnx import TensorProtoDataType
            from torch.onnx.symbolic_helper import _slice_helper
            from torch.onnx.symbolic_opset9 import squeeze, sub

            # batch_indices = rois[:, 0].long()
            batch_indices = _slice_helper(
                g, rois, axes=[1], starts=[0], ends=[1])
            batch_indices = squeeze(g, batch_indices, 1)
            batch_indices = g.op(
                'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
            # rois = rois[:, 1:]
            rois = _slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
            if aligned:
                # rois -= 0.5/spatial_scale
                aligned_offset = g.op(
                    'Constant',
                    value_t=torch.tensor([0.5 / spatial_scale],
                                         dtype=torch.float32))
                rois = sub(g, rois, aligned_offset)
            # roi align
            return g.op(
                'RoiAlignDeterministic',
                input,
                rois,
                batch_indices,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=max(0, sampling_ratio),
                mode_s=pool_mode)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('max', 'avg')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)

        ext_module.roi_align_forward(
            input,
            rois,
            output,
            argmax_y,
            argmax_x,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)

        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        #grad_output = grad_output * 10000000
        #grad_input2, _, _, _, _, _, _ = backward2(ctx, grad_output)
        '''
        rois, argmax_y, argmax_x = ctx.saved_tensors
        if ctx.pool_mode == 'max':
            sampling_ratio_max = 1
        else:
            if ctx.sampling_ratio > 0:
                sampling_ratio_max = ctx.sampling_ratio ** 2
            else:
                rois_height_max = torch.ceil((rois[:,4] - rois[:,2]).max() / ctx.output_size[0])
                rois_width_max = torch.ceil((rois[:,3] - rois[:,1]).max() / ctx.output_size[1])
                sampling_ratio_max = int(rois_height_max * rois_width_max)
        add_index = grad_output.new_zeros([*grad_output.shape, sampling_ratio_max * 4])
        add = grad_output.new_zeros([*grad_output.shape, sampling_ratio_max * 4])
        # complex head architecture may cause grad_output uncontiguous.
        add_index = add_index.view(-1).contiguous()
        add = add.view(-1).contiguous()
        grad_output = grad_output.contiguous()
        ext_module.roi_align_deterministic_backward(
            grad_output,
            rois,
            argmax_y,
            argmax_x,
            add_index, add,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned,
            channels=ctx.input_shape[1],
            height=ctx.input_shape[2],
            width=ctx.input_shape[3],
            sampling_ratio_max=sampling_ratio_max,
        )
        grad_input = grad_output.new_zeros(ctx.input_shape)
        grad_input = torch.index_add(grad_input.view(-1), 0, add_index.long(), add)
        #print(grad_output)
        #print(add)
        #print(add_index)
        grad_input = grad_input.view(*ctx.input_shape)
        return grad_input, None, None, None, None, None, None
        '''
        '''
        rois, argmax_y, argmax_x = ctx.saved_tensors
        add_index = grad_output.new_zeros(ctx.input_shape)
        add = grad_output.new_zeros([0])

        grad_output = grad_output.contiguous()
        ext_module.roi_align_deterministic_backward(
            grad_output,
            rois,
            argmax_y,
            argmax_x,
            add_index, add,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned,
            channels=ctx.input_shape[1],
            height=ctx.input_shape[2],
            width=ctx.input_shape[3],
            sampling_ratio_max=3,
        )
        grad_input = add_index

        return grad_input, None, None, None, None, None, None
        '''
        rois, argmax_y, argmax_x = ctx.saved_tensors
        if ctx.pool_mode == 0:
            add_index = grad_output.new_zeros([*grad_output.shape, 4]).view(-1).contiguous()
            add = grad_output.new_zeros([*grad_output.shape, 4]).view(-1).contiguous()
            grad_output = grad_output.contiguous()
            ext_module.roi_align_deterministic_backward(
                grad_output,
                rois,
                argmax_y,
                argmax_x,
                add_index, add,
                aligned_height=ctx.output_size[0],
                aligned_width=ctx.output_size[1],
                spatial_scale=ctx.spatial_scale,
                sampling_ratio=ctx.sampling_ratio,
                pool_mode=ctx.pool_mode,
                aligned=ctx.aligned,
                channels=ctx.input_shape[1],
                height=ctx.input_shape[2],
                width=ctx.input_shape[3],
                sampling_ratio_max=3,
            )
            grad_input = grad_output.new_zeros(ctx.input_shape)
            grad_input = torch.index_add(grad_input.view(-1), 0, add_index.long(), add).view(ctx.input_shape)
        else:
            # '''
            B, C, H, W = ctx.input_shape
            b, c, h, w = grad_output.shape
            # add = grad_output.new_zeros(ctx.input_shape)
            grad_output = grad_output.contiguous()
            small_batch = b // B
            # temp = grad_output.transpose(0, 1).reshape(C, b * h * w).T
            grad_output = torch.split(grad_output, small_batch, 0)
            rois = torch.split(rois, small_batch, 0)
            add = []
            for i, (g, r) in enumerate(zip(grad_output, rois)):
                add_index = g.new_zeros([small_batch * h * w * H * W, ])
                # print(add_index)
                ext_module.roi_align_deterministic_backward(
                    g,
                    r,
                    argmax_y,
                    argmax_x,
                    add_index, add_index,
                    aligned_height=ctx.output_size[0],
                    aligned_width=ctx.output_size[1],
                    spatial_scale=ctx.spatial_scale,
                    sampling_ratio=ctx.sampling_ratio,
                    pool_mode=ctx.pool_mode,
                    aligned=ctx.aligned,
                    batch_size=ctx.input_shape[0],
                    channels=ctx.input_shape[1],
                    height=ctx.input_shape[2],
                    width=ctx.input_shape[3],
                    sparse_index=g,
                )
                # print(add_index.view(small_batch, h * w, H * W))
                # print(add)
                grad_input = torch.mm(g.transpose(0, 1).reshape(c, small_batch * h * w),
                                      add_index.view(small_batch * h * w, H * W)).view(c, H, W)
                add.append(grad_input)
                del add_index, g
            grad_input = torch.stack(add, 0)
            # '''
            '''
            rois_height = torch.ceil((rois[:, 4] - rois[:, 2]) / ctx.output_size[0])
            rois_width = torch.ceil((rois[:, 3] - rois[:, 1]) / ctx.output_size[1])
            a = 4 * (rois_height * rois_width) * ctx.output_size[0] * ctx.output_size[1]
            sparse_index = torch.cumsum(a, 0)
            temp = sparse_index.new_zeros(a.shape)
            temp[1:] = sparse_index[:-1]
            sparse_dim = int(sparse_index[-1].item())
            sparse_index = temp.contiguous()
            #print(sparse_index)
            #print(sparse_dim)
            B, C, H, W = ctx.input_shape
            b, c, h, w = grad_output.shape
            grad_output = grad_output.contiguous()
            add_index = grad_output.new_zeros([sparse_dim, ])
            add = grad_output.new_zeros([sparse_dim, ])

            ext_module.roi_align_deterministic_backward(
                grad_output,
                rois,
                argmax_y,
                argmax_x,
                add_index, add,
                aligned_height=ctx.output_size[0],
                aligned_width=ctx.output_size[1],
                spatial_scale=ctx.spatial_scale,
                sampling_ratio=ctx.sampling_ratio,
                pool_mode=ctx.pool_mode,
                aligned=ctx.aligned,
                batch_size=ctx.input_shape[0],
                channels=ctx.input_shape[1],
                height=ctx.input_shape[2],
                width=ctx.input_shape[3],
                sparse_index=sparse_index,
            )
            x = torch.div(add_index, B * H * W, rounding_mode='trunc')
            y = add_index % (B * H * W)
            yx = torch.stack((y, x), 0).long()
            # yx_debug = yx.cpu().numpy()
            #print(add_index.max())
            s = torch.sparse_coo_tensor(yx, add, (B * H * W, b * h * w))
            #print(s)
            #print(grad_output)
            grad_input = torch.mm(s, grad_output.transpose(0, 1).reshape(C, b * h * w).T).T.view(C, B, H, W).transpose(
                0, 1)
            '''
            #print(((grad_input2-grad_input)/grad_input2).abs().mean(),(grad_input2).abs().mean())
        return grad_input, None, None, None, None, None, None


roi_align_deterministic = RoIAlignDeterministicFunction.apply


class RoIAlignDeterministic(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
        use_torchvision (bool): whether to use roi_align from torchvision.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    @deprecated_api_warning(
        {
            'out_size': 'output_size',
            'sample_num': 'sampling_ratio'
        },
        cls_name='RoIAlignDeterministic')
    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True):
        super(RoIAlignDeterministic, self).__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """

        return roi_align_deterministic(input, rois, self.output_size, self.spatial_scale,
                                       self.sampling_ratio, self.pool_mode, self.aligned)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned}, '
        return s
