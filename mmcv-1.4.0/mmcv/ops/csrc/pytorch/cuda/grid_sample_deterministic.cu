// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "grid_sample_deterministic_cuda_kernel.cuh"


void grid_sample_deterministic_forward_cuda(Tensor  input,Tensor  offset,Tensor  output,
                                      const int offset_num, const int channel,  const int height, const int width) {
  const int output_size = output.numel();
  at::cuda::CUDAGuard device_guard(output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(), "grid_sample_deterministic_forward_cuda_kernel", [&] {
        grid_sample_deterministic_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>( output_size,
            input.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            offset_num,channel,height,width);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void grid_sample_deterministic_input_backward_cuda(Tensor grad_output, Tensor t2_indice, Tensor offset, Tensor batch_id,
  Tensor t2_i_i_s_id, Tensor t2_i_i_f_id, Tensor grad_input, const int group,
  const int offset_num, const int channel, const int height, const int width) {
  const int output_size =  group * channel;
  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "grid_sample_deterministic_input_backward_cuda_kernel", [&] {
        grid_sample_deterministic_input_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>( output_size,
            grad_output.data_ptr<scalar_t>(),
            t2_indice.data_ptr<int64_t>(),
            offset.data_ptr<scalar_t>(),
            batch_id.data_ptr<int64_t>(),
            t2_i_i_s_id.data_ptr<int64_t>(),
            t2_i_i_f_id.data_ptr<int64_t>(),
            grad_input.data_ptr<scalar_t>(),
            group, offset_num,channel,height,width);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void grid_sample_deterministic_offset_backward_cuda(Tensor grad_output, Tensor input, Tensor offset, Tensor grad_offset,
  const int offset_num, const int channel,  const int height, const int width) {
  const int output_size =  grad_offset.numel() / 2;
  at::cuda::CUDAGuard device_guard(offset.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      offset.scalar_type(), "grid_sample_deterministic_offset_backward_cuda_kernel", [&] {
        grid_sample_deterministic_offset_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>( output_size,
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            grad_offset.data_ptr<scalar_t>(),
            offset_num,channel,height,width);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}



