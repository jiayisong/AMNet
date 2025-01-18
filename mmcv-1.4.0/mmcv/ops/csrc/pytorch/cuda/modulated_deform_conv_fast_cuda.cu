// Copyright (c) OpenMMLab. All rights reserved
#include "modulated_deform_conv_fast_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void modulated_deform_conv_fast_backward_cuda(Tensor grad, Tensor offset, Tensor input, Tensor  wh_mat, Tensor  offset_temp,
                                               const int channels_per_group,  const int height, const int width) {
  at::cuda::CUDAGuard device_guard(offset.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int num_kernels = height * width * channels_per_group;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      offset.scalar_type(), "modulated_deform_conv_fast_backward_cuda_kernel", [&] {
        modulated_deform_conv_fast_backward_cuda_kernel
        <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(

        num_kernels,
        grad.data_ptr<scalar_t>(), offset.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), wh_mat.data_ptr<scalar_t>(), offset_temp.data_ptr<scalar_t>(),
         channels_per_group, height, width

        );
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void modulated_deform_conv_fast_forward_cuda(Tensor offset, Tensor input, Tensor output,
                                               const int kernel_h, const int kernel_w, const int batch_size, const int deform_groups,
                                               const int channels_per_group,  const int height, const int width) {
  at::cuda::CUDAGuard device_guard(offset.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int num_kernels = batch_size * deform_groups * kernel_h * kernel_w * channels_per_group * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "modulated_deform_conv_fast_forward_cuda_kernel", [&] {

        modulated_deform_conv_fast_forward_cuda_kernel
        <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels,
          offset.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          kernel_h, kernel_w, batch_size, deform_groups, channels_per_group,  height, width

          );
      });
  AT_CUDA_CHECK(cudaGetLastError());
}