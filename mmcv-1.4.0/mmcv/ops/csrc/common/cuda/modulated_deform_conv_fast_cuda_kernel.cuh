
#ifndef MODULATED_DEFORM_CONV_FAST_CUDA_KERNEL_CUH
#define MODULATED_DEFORM_CONV_FAST_CUDA_KERNEL_CUH

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
__global__ void modulated_deform_conv_fast_forward_cuda_kernel(const int n,
    const T* offset, const T* input, T* output,
    const int kernel_h, const int kernel_w, const int batch_size, const int deform_groups,
    const int channels_per_group,  const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_id = index % width;
    const int h_id = (index / width) % height;
    const int c_id = (index / width / height) % channels_per_group;
    const int kw_id = (index / width / height / channels_per_group) % kernel_w;
    const int kh_id = (index / width / height / channels_per_group / kernel_w) % kernel_h;
    const int g_id = (index / width / height / channels_per_group / kernel_w / kernel_h) % deform_groups;
    const int b_id = (index / width / height / channels_per_group / kernel_w / kernel_h / deform_groups);

    const T *input_ptr = input +
        ((b_id * deform_groups + g_id) * channels_per_group + c_id) * height * width;
    const T *offset_x_ptr = offset +
        (((((b_id * deform_groups + g_id) * kernel_h + kh_id )* kernel_w + kw_id) * 2 + 0) * height + h_id) * width + w_id;
    const T *offset_y_ptr = offset +
        (((((b_id * deform_groups + g_id) * kernel_h + kh_id )* kernel_w + kw_id) * 2 + 1) * height + h_id) * width + w_id;
    T y = h_id + offset_x_ptr[0];
    T x = w_id + offset_y_ptr[0];
    output[index] = bilinear_interpolate(input_ptr, height, width, y, x, 0);
  }
}

template <typename T>
__global__ void modulated_deform_conv_fast_backward_cuda_kernel(
    const int n, const T* grad, const T* offset, const T*input, T* wh_mat, T* offset_temp,const int channels_per_group, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, n) {
        const int w_out = index % width;
        const int h_out = (index / width) & height ;
        const int c_out = index / width / height ;
        const int hw_out =  (h_out * width + w_out);
        const T offset_h = offset[hw_out+width*height];
        const T offset_w = offset[hw_out];
        const T cur_inv_h_data = h_out + offset_h;
        const T cur_inv_w_data = w_out + offset_w;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, cur_inv_h_data, cur_inv_w_data, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, 0);
        const int cur_input = c_out * width * height;
        T v1,v2,v3,v4;
        if (y_low < 0 || x_low < 0) {
            v1 = 0;
        }else{
            v1 = input[cur_input+y_low * width + x_low];
        }
        if (y_low < 0 || x_high < 0) {
            v2 = 0;
        }else{
            v2 = input[cur_input+y_low * width + x_high];
        }
        if (y_high < 0 || x_low < 0) {
            v3 = 0;
        }else{
            v3 = input[cur_input+y_high * width + x_low];
        }
        if (y_high < 0 || x_high < 0) {
            v4 = 0;
        }else{
            v4 = input[cur_input+y_high * width + x_high];
        }
        offset_temp[index] = grad[index]*((cur_inv_h_data)*(v4-v3)+(-cur_inv_h_data)*(v2-v1));
        offset_temp[index + channels_per_group*width*height] = grad[index]*((cur_inv_w_data-x_low)*(v4-v2)+(x_high-cur_inv_w_data)*(v3-v1));

        if (c_out==0){
        const int cur_bottom_grad_pos = hw_out * height * width;
        if (y_low >= 0 && x_low >= 0) {
            wh_mat[cur_bottom_grad_pos + y_low * width + x_low] = w1;
        }
        if (y_low >= 0 && x_high >= 0) {
            wh_mat[cur_bottom_grad_pos + y_low * width + x_high] = w2;
        }
        if (y_high >= 0 && x_low >= 0) {
            wh_mat[cur_bottom_grad_pos + y_high * width + x_low] = w3;
        }
        if (y_high >= 0 && x_high >= 0) {
            wh_mat[cur_bottom_grad_pos + y_high * width + x_high] = w4;
        }
        }

    }
}


#endif // MODULATED_DEFORM_CONV_FAST_CUDA_KERNEL_CUH
