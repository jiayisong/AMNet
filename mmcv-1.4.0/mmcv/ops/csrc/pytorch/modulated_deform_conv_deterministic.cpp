// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
//#include "modulated_deform_conv.cpp"

void modulated_deformable_im2col_impl(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_coord_impl(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);
void modulated_deformable_col2im_deterministic_impl(const Tensor data_offset, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int kh_id, const int kw_id,const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int g_id, Tensor add_index, Tensor add_value) {
  DISPATCH_DEVICE_IMPL(modulated_deformable_col2im_deterministic_impl, data_offset,
                       height_im, width_im,
                       height_col, width_col, kernel_h, kernel_w, kh_id, kw_id, pad_h, pad_w,
                       stride_h, stride_w, dilation_h, dilation_w,
                       g_id, add_index, add_value);
}

void modulated_deform_conv_deterministic_forward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias, const int im2col_step) {
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  // resize output
  output = output.view({batch, channels_out, height_out, width_out}).zero_();
  // resize temporary columns
  columns =
      at::zeros({channels * kernel_h * kernel_w, im2col_step * height_out * width_out},
                input.options());

  output = output.view({batch / im2col_step, group, output.size(1) / group,
                         im2col_step * output.size(2), output.size(3)});
  input = input.view({batch / im2col_step, im2col_step, channels, height, width});
  offset = offset.view({batch / im2col_step, im2col_step, -1});
  mask = mask.view({batch / im2col_step, im2col_step, -1});
  for (int b = 0; b < batch / im2col_step; b++) {
    modulated_deformable_im2col_impl(
        input[b], offset[b], mask[b], im2col_step, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);

    // divide into group
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});

    for (int g = 0; g < group; g++) {
      output[b][g] = output[b][g]
                         .flatten(1)
                         .addmm_(weight[g].flatten(1), columns[g])
                         .view_as(output[b][g]);
    }

    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

output = output.view({batch / im2col_step, channels_out,
                                      im2col_step, height_out, width_out});
  output.transpose_(1, 2);

  output = output.view({batch, channels_out, height_out, width_out});


  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }
}


void modulated_deform_conv_deterministic_backward(
    Tensor input, Tensor offset,
    Tensor mask, Tensor columns, Tensor add_index, Tensor add_value,
    Tensor grad_offset, Tensor grad_mask,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int deformable_group) {
  at::DeviceGuard guard(input.device());

  const int channels = input.size(0);
  const int height = input.size(1);
  const int width = input.size(2);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;



    // gradient w.r.t. input coordinate data
    modulated_deformable_col2im_coord_impl(
        columns, input, offset, mask, 1, channels, height, width,
        height_out, width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, dilation_h, dilation_w, deformable_group, grad_offset,
        grad_mask);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and
    // group
    modulated_deformable_im2col_impl(
        input, offset, mask, 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);

}

void modulated_deformable_col2im_deterministic(
    Tensor offset, Tensor add_index, Tensor add_value, const int height,const int width,
    const int height_out, const int width_out, const int kh_id, const int kw_id,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w, const int g_id) {
  at::DeviceGuard guard(offset.device());

    modulated_deformable_col2im_deterministic_impl(
         offset, height, width, height_out,
        width_out, kernel_h, kernel_w, kh_id, kw_id, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, g_id, add_index, add_value);

}
