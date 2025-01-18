// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"


void grid_sample_deterministic_forward_impl( Tensor  input,Tensor  offset,Tensor  output,
                                      const int offset_num, const int channel,  const int height, const int width) {
  DISPATCH_DEVICE_IMPL(grid_sample_deterministic_forward_impl,  input, offset, output,offset_num,channel,height,width);
}

void grid_sample_deterministic_input_backward_impl( Tensor grad_output, Tensor t2_indice, Tensor offset, Tensor batch_id,
  Tensor t2_i_i_s_id, Tensor t2_i_i_f_id, Tensor grad_input, const int group,
  const int offset_num, const int channel, const int height, const int width) {
  DISPATCH_DEVICE_IMPL(grid_sample_deterministic_input_backward_impl,  grad_output, t2_indice, offset, batch_id, t2_i_i_s_id,
                                                            t2_i_i_f_id, grad_input, group, offset_num,channel,height,width);
}
void grid_sample_deterministic_offset_backward_impl(Tensor grad_output, Tensor input, Tensor offset, Tensor grad_offset,
  const int offset_num, const int channel,  const int height, const int width) {
  DISPATCH_DEVICE_IMPL(grid_sample_deterministic_offset_backward_impl,  grad_output, input, offset, grad_offset, offset_num,
                                                            channel, height, width);
}


void grid_sample_deterministic_forward(Tensor  input,Tensor  offset,Tensor  output,
                                      const int offset_num, const int channel,  const int height, const int width) {
grid_sample_deterministic_forward_impl( input, offset, output,offset_num,channel,height,width);
}

void grid_sample_deterministic_input_backward(Tensor grad_output, Tensor t2_indice, Tensor offset, Tensor batch_id,
  Tensor t2_i_i_s_id, Tensor t2_i_i_f_id, Tensor grad_input, const int group,
  const int offset_num, const int channel, const int height, const int width) {
grid_sample_deterministic_input_backward_impl(grad_output, t2_indice, offset, batch_id, t2_i_i_s_id,
                                                            t2_i_i_f_id, grad_input, group, offset_num,channel,height,width);
}
void grid_sample_deterministic_offset_backward(Tensor grad_output, Tensor input, Tensor offset, Tensor grad_offset,
  const int offset_num, const int channel,  const int height, const int width) {
grid_sample_deterministic_offset_backward_impl(grad_output, input, offset, grad_offset, offset_num,
                                                            channel, height, width);
}