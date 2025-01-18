// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "roi_align_deterministic_cuda_kernel.cuh"


/*
void ROIAlignDeterministicBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor argmax_y, Tensor argmax_x,
                                        Tensor index_add, Tensor add, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio, int pool_mode,
                                        bool aligned, int channels, int height, int width, int sampling_ratio_max) {
  int output_size = grad_output.numel();

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "roi_align_deterministic_backward_cuda_kernel", [&] {
        roi_align_deterministic_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
                argmax_x.data_ptr<scalar_t>(), index_add.data_ptr<scalar_t>(),add.data_ptr<scalar_t>(),
                aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
                aligned, channels, height, width, sampling_ratio_max);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}
*/
void ROIAlignDeterministicBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor argmax_y, Tensor argmax_x,
                                        Tensor index_add, Tensor add, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio, int pool_mode,
                                        bool aligned,int batch_size, int channels, int height, int width, Tensor sparse_index) {

  int output_size = 0;
  if (pool_mode==0){
    output_size = grad_output.numel();
  }
  else{
    output_size = grad_output.size(0) * grad_output.size(2) * grad_output.size(3);
  }
  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "roi_align_deterministic_backward_cuda_kernel", [&] {
        roi_align_deterministic_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
                argmax_x.data_ptr<scalar_t>(), index_add.data_ptr<scalar_t>(),add.data_ptr<scalar_t>(),
                aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
                aligned,batch_size, channels, height, width, sparse_index.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}
