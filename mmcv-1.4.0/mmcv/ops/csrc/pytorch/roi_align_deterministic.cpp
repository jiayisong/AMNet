// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"


void roi_align_deterministic_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor index_add, Tensor add,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned,int batch_size,  int channels, int height, int width, Tensor sparse_index) {
  DISPATCH_DEVICE_IMPL(roi_align_deterministic_backward_impl, grad_output, rois, argmax_y,
                       argmax_x, index_add, add, aligned_height, aligned_width,
                       spatial_scale, sampling_ratio, pool_mode, aligned, batch_size, channels, height, width, sparse_index);
}

void roi_align_deterministic_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor index_add, Tensor add, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned, int batch_size, int channels, int height, int width, Tensor sparse_index) {
  roi_align_deterministic_backward_impl(grad_output, rois, argmax_y, argmax_x, index_add, add,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned, batch_size, channels, height, width, sparse_index);
}
