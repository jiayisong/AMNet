#ifndef COMMON_CUDA_HELPER
#define COMMON_CUDA_HELPER

#include <cuda.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y <= -1.0 || y >= height || x <= -1.0 || x >= width) return 0;

  int y_low = floorf(y);
  int x_low = floorf(x);
  int y_high = y_low + 1;
  int x_high = x_low + 1;

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1, v2, v3, v4;
  if (y_low < 0 || x_low < 0) {
    v1 = 0;
  }else{
    v1 = input[y_low * width + x_low];
  }
  if (y_low < 0 || x_high > (width - 1)) {
    v2 = 0;
  }else{
    v2 = input[y_low * width + x_high];
  }
  if (y_high > (height - 1) || x_low < 0) {
    v3 = 0;
  }else{
    v3 = input[y_high * width + x_low];
  }
  if (y_high > (height - 1) || x_high > (width - 1)) {
    v4 = 0;
  }else{
    v4 = input[y_high * width + x_high];
  }
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y <= -1.0 || y >= height || x <= -1.0 || x >= width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }
  y_low = floorf(y);
  x_low = floorf(x);
  y_high = y_low + 1;
  x_high = x_low + 1;

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  if (x_high > (width - 1)) {
    x_high = -1;
  }
  if (y_high > (height - 1)) {
    y_high = -1;
  }
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}
#endif  // COMMON_CUDA_HELPER
