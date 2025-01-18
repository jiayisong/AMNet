// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"


void modulated_deform_conv_fast_forward_impl(
   Tensor offset, Tensor input, Tensor output,
   const int kernel_h, const int kernel_w, const int batch_size, const int deform_groups,
   const int channels_per_group,  const int height, const int width) {
  DISPATCH_DEVICE_IMPL(modulated_deform_conv_fast_forward_impl,  offset, input, output,
          kernel_h, kernel_w, batch_size, deform_groups, channels_per_group,  height, width);
}

void modulated_deform_conv_fast_backward_impl(Tensor grad, Tensor offset, Tensor input, Tensor  wh_mat, Tensor  offset_temp,
                                               const int channels_per_group,  const int height, const int width) {
  DISPATCH_DEVICE_IMPL(modulated_deform_conv_fast_backward_impl,  grad, offset, input, wh_mat, offset_temp,
         channels_per_group, height, width);
}

void modulated_deform_conv_fast_forward(
    Tensor offset, Tensor input, Tensor output,
   const int kernel_h, const int kernel_w, const int batch_size, const int deform_groups,
   const int channels_per_group,  const int height, const int width) {
modulated_deform_conv_fast_forward_impl( offset, input, output,
          kernel_h, kernel_w, batch_size, deform_groups, channels_per_group,  height, width);
}

void modulated_deform_conv_fast_backward(Tensor grad, Tensor offset, Tensor input, Tensor  wh_mat, Tensor  offset_temp,
                                               const int channels_per_group,  const int height, const int width) {
modulated_deform_conv_fast_backward_impl(grad, offset, input, wh_mat, offset_temp,
         channels_per_group, height, width);
}
