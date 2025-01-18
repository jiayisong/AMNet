
#ifndef GRID_SAMPLE_DETERMINISTIC_CUDA_KERNEL_CUH
#define GRID_SAMPLE_DETERMINISTIC_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT




template <typename T>
__global__ void grid_sample_deterministic_forward_cuda_kernel(const int n,
     const T* input, const T* offset, T* output, const int offset_num,
    const int channel,  const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int n_id = index % offset_num;
    const int c_id = (index / offset_num) % channel;
    const int b_id = index / offset_num / channel;

    const T *input_ptr = input + (b_id * channel + c_id) * height * width;
    const T *offset_x_ptr = offset + (b_id * 2 + 0) * offset_num + n_id;
    const T *offset_y_ptr = offset + (b_id * 2 + 1) * offset_num + n_id;
    T x = offset_x_ptr[0];
    T y = offset_y_ptr[0];
    output[index] = bilinear_interpolate(input_ptr, height, width, y, x, 0);
  }
}

template <typename T>
__global__ void grid_sample_deterministic_input_backward_cuda_kernel(const int n,
  const T* grad_output, const long* t2_indice, const T* offset, const long* batch_id,
  const long* t2_i_i_s_id, const long* t2_i_i_f_id, T* grad_input, const int group,
  const int offset_num, const int channel, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, n) {
        const int ind = index % group;
        const int c_id = index / group;
        const int b_id = batch_id[ind];

        T *grad_input_ptr = grad_input + (b_id * channel + c_id) * height * width;
        const T *offset_x_ptr = offset + (b_id * 2 + 0) * offset_num;
        const T *offset_y_ptr = offset + (b_id * 2 + 1) * offset_num;
        const T *grad_output_ptr = grad_output + (b_id * channel + c_id) * offset_num;
        const long *t2_indice_ptr = t2_indice + b_id * offset_num;
        const int t2_s = t2_i_i_s_id[ind];
        int t2_f = t2_i_i_f_id[ind];
        if (t2_f == 0){
            t2_f = offset_num;
        }
        for (int i=t2_s; i<t2_f; i++){
            const int n_id = t2_indice_ptr[i];
            T x = offset_x_ptr[n_id];
            T y = offset_y_ptr[n_id];
            T grad = grad_output_ptr[n_id];
            T w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;
            bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high, 0);

            if (y_low >= 0 && x_low >= 0) {
                grad_input_ptr[y_low * width + x_low] += w1 * grad;
            }
            if (y_low >= 0 && x_high >= 0) {
                grad_input_ptr[y_low * width + x_high] += w2 * grad;
            }
            if (y_high >= 0 && x_low >= 0) {
                grad_input_ptr[y_high * width + x_low] += w3 * grad;
            }
            if (y_high >= 0 && x_high >= 0) {
                grad_input_ptr[y_high * width + x_high] += w4 * grad;
            }

        }



    }
}


template <typename T>
__global__ void grid_sample_deterministic_offset_backward_cuda_kernel(const int n,
  const T* grad_output, const T* input, const T* offset, T* grad_offset,
  const int offset_num, const int channel,  const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, n) {
        const int n_id = index % offset_num;
        const int b_id = index / offset_num;

        const T *input_ptr = input + b_id * channel * height * width;
        const T *offset_x_ptr = offset + (b_id * 2 + 0) * offset_num + n_id;
        const T *offset_y_ptr = offset + (b_id * 2 + 1) * offset_num + n_id;
        const T *grad_output_ptr = grad_output + b_id * channel * offset_num + n_id;
        T *grad_offset_x_ptr = grad_offset + (b_id * 2 + 0) * offset_num + n_id;
        T *grad_offset_y_ptr = grad_offset + (b_id * 2 + 1) * offset_num + n_id;
        T x = offset_x_ptr[0];
        T y = offset_y_ptr[0];

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        T v1,v2,v3,v4;
        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high, 0);
        T Ox1 = x - x_low;
        T Oy1 = y - y_low;
        T Ox2 = 1 - Ox1;
        T Oy2 = 1 - Oy1;
        T x_result = 0;
        T y_result = 0;
        for (int c_id=0; c_id<channel; c_id++){
            const T *input_ptr_c = input_ptr + c_id * height * width;
            const T *grad_output_ptr_c = grad_output_ptr + c_id * offset_num;

            if (y_low < 0 || x_low < 0) {
                v1 = 0;
            }else{
                v1 = input_ptr_c[y_low * width + x_low];
            }
            if (y_low < 0 || x_high < 0) {
                v2 = 0;
            }else{
                v2 = input_ptr_c[y_low * width + x_high];
            }
            if (y_high < 0 || x_low < 0) {
                v3 = 0;
            }else{
                v3 = input_ptr_c[y_high * width + x_low];
            }
            if (y_high < 0 || x_high < 0) {
                v4 = 0;
            }else{
                v4 = input_ptr_c[y_high * width + x_high];
            }
            x_result += (grad_output_ptr_c[0]*(Oy1*(v4-v3)+Oy2*(v2-v1)));
            y_result += (grad_output_ptr_c[0]*(Ox1*(v4-v2)+Ox2*(v3-v1)));
        }
        grad_offset_x_ptr[0] = x_result;
        grad_offset_y_ptr[0] = y_result;
    }
}


#endif // GRID_SAMPLE_DETERMINISTIC_CUDA_KERNEL_CUH
