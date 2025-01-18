// Copyright (c) OpenMMLab. All rights reserved
#include "modulated_deform_conv_deterministic_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"


void modulated_deformable_col2im_deterministic_cuda(
   const Tensor data_offset, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int kh_id, const int kw_id,const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int g_id, Tensor add_index, Tensor add_value) {
  const int num_kernels =  height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_offset.scalar_type(), "modulated_deformable_col2im_deterministic_gpu", ([&] {
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *add_value_ = add_value.data_ptr<scalar_t>();
        scalar_t *add_index_ = add_index.data_ptr<scalar_t>();

        modulated_deformable_col2im_deterministic_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels,  data_offset_,
            height_im, width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, g_id, kh_id, kw_id,
            height_col, width_col, add_index_, add_value_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}