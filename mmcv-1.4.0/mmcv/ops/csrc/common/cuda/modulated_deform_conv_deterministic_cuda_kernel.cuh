
#ifndef MODULATED_DEFORM_CONV_DETERMINISTIC_CUDA_KERNEL_CUH
#define MODULATED_DEFORM_CONV_DETERMINISTIC_CUDA_KERNEL_CUH

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
__global__ void modulated_deformable_col2im_deterministic_gpu_kernel(
    const int n, const T *data_offset,
    const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int g_id, const int i, const int j ,
    const int height_col, const int width_col,
    T *add_index, T *add_value) {
  CUDA_1D_KERNEL_LOOP(index, n) {
        const int w_out = index % width_col;
        const int h_out = (index / width_col) % height_col;
        const int w_in = w_out * stride_w - pad_w;
        const int h_in = h_out * stride_h - pad_h;

        const T *data_offset_ptr =
                data_offset + ( g_id) * 2 *
                              kernel_h * kernel_w * height_col * width_col;
        //const T *data_mask_ptr =
        //        data_mask + (g_id) * kernel_h *
        //                    kernel_w * height_col * width_col;
        const int data_offset_h_ptr =
                ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
        const int data_offset_w_ptr =
                ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
        //const int data_mask_hw_ptr =
        //        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        //const T mask = data_mask_ptr[data_mask_hw_ptr];
        const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
        const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

        //const T cur_top_grad = data_col[index] * mask;
        //const int cur_h = (int) cur_inv_h_data;
        //const int cur_w = (int) cur_inv_w_data;
        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, cur_inv_h_data, cur_inv_w_data, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, 0);
        const int cur_bottom_grad_pos = index * height * width;
        if (y_low >= 0 && x_low >= 0) {
            //add_index[index*4] = cur_bottom_grad_pos + y_low * width + x_low;
            add_value[cur_bottom_grad_pos + y_low * width + x_low] = w1;
        }
        if (y_low >= 0 && x_high >= 0) {
            //add_index[index*4+1] = cur_bottom_grad_pos + y_low * width + x_high;
            add_value[cur_bottom_grad_pos + y_low * width + x_high] = w2;
        }
        if (y_high >= 0 && x_low >= 0) {
            //add_index[index*4+2] = cur_bottom_grad_pos + y_high * width + x_low;
            add_value[cur_bottom_grad_pos + y_high * width + x_low] = w3;
        }
        if (y_high >= 0 && x_high >= 0) {
            //add_index[index*4+3] = cur_bottom_grad_pos + y_high * width + x_high;
            add_value[cur_bottom_grad_pos + y_high * width + x_high] = w4;
        }

    }
}

#endif  // MODULATED_DEFORM_CONV_DETERMINISTIC_CUDA_KERNEL_CUH