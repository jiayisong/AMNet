#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void AssignScoreWithKForwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& points, const Tensor& centers, const Tensor& scores,
    const Tensor& knn_idx, Tensor& output);

void AssignScoreWithKBackwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores);

void assign_score_withk_forward_cuda(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor& points,
                                     const Tensor& centers,
                                     const Tensor& scores,
                                     const Tensor& knn_idx, Tensor& output) {
  AssignScoreWithKForwardCUDAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, points, centers, scores, knn_idx, output);
};

void assign_score_withk_backward_cuda(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores) {
  AssignScoreWithKBackwardCUDAKernelLauncher(
      B, N0, N1, M, K, O, aggregate, grad_out, points, centers, scores, knn_idx,
      grad_points, grad_centers, grad_scores);
};

void assign_score_withk_forward_impl(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor& points,
                                     const Tensor& centers,
                                     const Tensor& scores,
                                     const Tensor& knn_idx, Tensor& output);

void assign_score_withk_backward_impl(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores);

REGISTER_DEVICE_IMPL(assign_score_withk_forward_impl, CUDA,
                     assign_score_withk_forward_cuda);
REGISTER_DEVICE_IMPL(assign_score_withk_backward_impl, CUDA,
                     assign_score_withk_backward_cuda);

void BallQueryForwardCUDAKernelLauncher(int b, int n, int m, float min_radius,
                                        float max_radius, int nsample,
                                        const Tensor new_xyz, const Tensor xyz,
                                        Tensor idx);

void ball_query_forward_cuda(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx) {
  BallQueryForwardCUDAKernelLauncher(b, n, m, min_radius, max_radius, nsample,
                                     new_xyz, xyz, idx);
};

void ball_query_forward_impl(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx);
REGISTER_DEVICE_IMPL(ball_query_forward_impl, CUDA, ball_query_forward_cuda);

void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset);

void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  BBoxOverlapsCUDAKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);
REGISTER_DEVICE_IMPL(bbox_overlaps_impl, CUDA, bbox_overlaps_cuda);

void BorderAlignForwardCUDAKernelLauncher(const Tensor& input,
                                          const Tensor& boxes, Tensor output,
                                          Tensor argmax_idx,
                                          const int pool_size);

void BorderAlignBackwardCUDAKernelLauncher(const Tensor& grad_output,
                                           const Tensor& boxes,
                                           const Tensor& argmax_idx,
                                           Tensor grad_input,
                                           const int pool_size);

void border_align_forward_cuda(const Tensor& input, const Tensor& boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size) {
  BorderAlignForwardCUDAKernelLauncher(input, boxes, output, argmax_idx,
                                       pool_size);
}

void border_align_backward_cuda(const Tensor& grad_output, const Tensor& boxes,
                                const Tensor& argmax_idx, Tensor grad_input,
                                const int pool_size) {
  BorderAlignBackwardCUDAKernelLauncher(grad_output, boxes, argmax_idx,
                                        grad_input, pool_size);
}

void border_align_forward_impl(const Tensor& input, const Tensor& boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size);

void border_align_backward_impl(const Tensor& grad_output, const Tensor& boxes,
                                const Tensor& argmax_idx, Tensor grad_input,
                                const int pool_size);

REGISTER_DEVICE_IMPL(border_align_forward_impl, CUDA,
                     border_align_forward_cuda);
REGISTER_DEVICE_IMPL(border_align_backward_impl, CUDA,
                     border_align_backward_cuda);

void box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);

void box_iou_rotated_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);
REGISTER_DEVICE_IMPL(box_iou_rotated_impl, CUDA, box_iou_rotated_cuda);

void CARAFEForwardCUDAKernelLauncher(const Tensor features, const Tensor masks,
                                     Tensor rfeatures, Tensor routput,
                                     Tensor rmasks, Tensor output,
                                     const int kernel_size,
                                     const int group_size,
                                     const int scale_factor);

void CARAFEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor rfeatures, const Tensor masks,
    Tensor rtop_grad, Tensor rbottom_grad_hs, Tensor rbottom_grad,
    Tensor rmask_grad, Tensor bottom_grad, Tensor mask_grad,
    const int kernel_size, const int group_size, const int scale_factor);

void carafe_forward_cuda(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor) {
  CARAFEForwardCUDAKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                  output, kernel_size, group_size,
                                  scale_factor);
}

void carafe_backward_cuda(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor) {
  CARAFEBackwardCUDAKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                   rbottom_grad_hs, rbottom_grad, rmask_grad,
                                   bottom_grad, mask_grad, kernel_size,
                                   group_size, scale_factor);
}

void carafe_forward_impl(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor);

void carafe_backward_impl(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor);

REGISTER_DEVICE_IMPL(carafe_forward_impl, CUDA, carafe_forward_cuda);
REGISTER_DEVICE_IMPL(carafe_backward_impl, CUDA, carafe_backward_cuda);

void CARAFENAIVEForwardCUDAKernelLauncher(const Tensor features,
                                          const Tensor masks, Tensor output,
                                          const int kernel_size,
                                          const int group_size,
                                          const int scale_factor);

void CARAFENAIVEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor features, const Tensor masks,
    Tensor bottom_grad, Tensor mask_grad, const int kernel_size,
    const int group_size, const int scale_factor);

void carafe_naive_forward_cuda(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor) {
  CARAFENAIVEForwardCUDAKernelLauncher(features, masks, output, kernel_size,
                                       group_size, scale_factor);
}

void carafe_naive_backward_cuda(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor) {
  CARAFENAIVEBackwardCUDAKernelLauncher(top_grad, features, masks, bottom_grad,
                                        mask_grad, kernel_size, group_size,
                                        scale_factor);
}
void carafe_naive_forward_impl(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor);

void carafe_naive_backward_impl(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor);

REGISTER_DEVICE_IMPL(carafe_naive_forward_impl, CUDA,
                     carafe_naive_forward_cuda);
REGISTER_DEVICE_IMPL(carafe_naive_backward_impl, CUDA,
                     carafe_naive_backward_cuda);

void CorrelationForwardCUDAKernelLauncher(Tensor input1, Tensor input2,
                                          Tensor output, int kH, int kW,
                                          int patchH, int patchW, int padH,
                                          int padW, int dilationH,
                                          int dilationW, int dilation_patchH,
                                          int dilation_patchW, int dH, int dW);

void CorrelationBackwardCUDAKernelLauncher(Tensor grad_output, Tensor input1,
                                           Tensor input2, Tensor grad_input1,
                                           Tensor grad_input2, int kH, int kW,
                                           int patchH, int patchW, int padH,
                                           int padW, int dilationH,
                                           int dilationW, int dilation_patchH,
                                           int dilation_patchW, int dH, int dW);

void correlation_forward_cuda(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW) {
  CorrelationForwardCUDAKernelLauncher(
      input1, input2, output, kH, kW, patchH, patchW, padH, padW, dilationH,
      dilationW, dilation_patchH, dilation_patchW, dH, dW);
}

void correlation_backward_cuda(Tensor grad_output, Tensor input1, Tensor input2,
                               Tensor grad_input1, Tensor grad_input2, int kH,
                               int kW, int patchH, int patchW, int padH,
                               int padW, int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW, int dH,
                               int dW) {
  CorrelationBackwardCUDAKernelLauncher(
      grad_output, input1, input2, grad_input1, grad_input2, kH, kW, patchH,
      patchW, padH, padW, dilationH, dilationW, dilation_patchH,
      dilation_patchW, dH, dW);
}

void correlation_forward_impl(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW);

void correlation_backward_impl(Tensor grad_output, Tensor input1, Tensor input2,
                               Tensor grad_input1, Tensor grad_input2, int kH,
                               int kW, int patchH, int patchW, int padH,
                               int padW, int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW, int dH,
                               int dW);

REGISTER_DEVICE_IMPL(correlation_forward_impl, CUDA, correlation_forward_cuda);
REGISTER_DEVICE_IMPL(correlation_backward_impl, CUDA,
                     correlation_backward_cuda);

void deformable_im2col_cuda(Tensor data_im, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor data_col);

void deformable_col2im_cuda(Tensor data_col, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor grad_im);

void deformable_col2im_coord_cuda(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

void deformable_im2col_impl(Tensor data_im, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor data_col);

void deformable_col2im_impl(Tensor data_col, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor grad_im);

void deformable_col2im_coord_impl(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

REGISTER_DEVICE_IMPL(deformable_im2col_impl, CUDA, deformable_im2col_cuda);
REGISTER_DEVICE_IMPL(deformable_col2im_impl, CUDA, deformable_col2im_cuda);
REGISTER_DEVICE_IMPL(deformable_col2im_coord_impl, CUDA,
                     deformable_col2im_coord_cuda);

void DeformRoIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                            Tensor offset, Tensor output,
                                            int pooled_height, int pooled_width,
                                            float spatial_scale,
                                            int sampling_ratio, float gamma);

void DeformRoIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma);

void deform_roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DeformRoIPoolForwardCUDAKernelLauncher(input, rois, offset, output,
                                         pooled_height, pooled_width,
                                         spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_backward_cuda(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma) {
  DeformRoIPoolBackwardCUDAKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_forward_impl(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma);

void deform_roi_pool_backward_impl(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma);

REGISTER_DEVICE_IMPL(deform_roi_pool_forward_impl, CUDA,
                     deform_roi_pool_forward_cuda);
REGISTER_DEVICE_IMPL(deform_roi_pool_backward_impl, CUDA,
                     deform_roi_pool_backward_cuda);

void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void sigmoid_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void sigmoid_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha) {
  SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                             gamma, alpha);
}

void softmax_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SoftmaxFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void softmax_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha) {
  SoftmaxFocalLossBackwardCUDAKernelLauncher(input, target, weight, buff,
                                             grad_input, gamma, alpha);
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, CUDA,
                     sigmoid_focal_loss_forward_cuda);
REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, CUDA,
                     sigmoid_focal_loss_backward_cuda);
REGISTER_DEVICE_IMPL(softmax_focal_loss_forward_impl, CUDA,
                     softmax_focal_loss_forward_cuda);
REGISTER_DEVICE_IMPL(softmax_focal_loss_backward_impl, CUDA,
                     softmax_focal_loss_backward_cuda);

void FurthestPointSamplingForwardCUDAKernelLauncher(int b, int n, int m,
                                                    const float* dataset,
                                                    float* temp, int* idxs);

void FurthestPointSamplingWithDistForwardCUDAKernelLauncher(
    int b, int n, int m, const float* dataset, float* temp, int* idxs);

void furthest_point_sampling_forward_cuda(Tensor points_tensor,
                                          Tensor temp_tensor, Tensor idx_tensor,
                                          int b, int n, int m) {
  const float* dataset = points_tensor.data_ptr<float>();
  float* temp = temp_tensor.data_ptr<float>();
  int* idxs = idx_tensor.data_ptr<int>();
  FurthestPointSamplingForwardCUDAKernelLauncher(b, n, m, dataset, temp, idxs);
}

void furthest_point_sampling_with_dist_forward_cuda(Tensor points_tensor,
                                                    Tensor temp_tensor,
                                                    Tensor idx_tensor, int b,
                                                    int n, int m) {
  const float* dataset = points_tensor.data_ptr<float>();
  float* temp = temp_tensor.data_ptr<float>();
  int* idxs = idx_tensor.data_ptr<int>();
  FurthestPointSamplingWithDistForwardCUDAKernelLauncher(b, n, m, dataset, temp,
                                                         idxs);
}

void furthest_point_sampling_forward_impl(Tensor points_tensor,
                                          Tensor temp_tensor, Tensor idx_tensor,
                                          int b, int n, int m);

void furthest_point_sampling_with_dist_forward_impl(Tensor points_tensor,
                                                    Tensor temp_tensor,
                                                    Tensor idx_tensor, int b,
                                                    int n, int m);

REGISTER_DEVICE_IMPL(furthest_point_sampling_forward_impl, CUDA,
                     furthest_point_sampling_forward_cuda);
REGISTER_DEVICE_IMPL(furthest_point_sampling_with_dist_forward_impl, CUDA,
                     furthest_point_sampling_with_dist_forward_cuda);

torch::Tensor fused_bias_leakyrelu_op(const torch::Tensor& input,
                                      const torch::Tensor& bias,
                                      const torch::Tensor& refer, int act,
                                      int grad, float alpha, float scale);

torch::Tensor fused_bias_leakyrelu_op_impl(const torch::Tensor& input,
                                           const torch::Tensor& bias,
                                           const torch::Tensor& refer, int act,
                                           int grad, float alpha, float scale);
REGISTER_DEVICE_IMPL(fused_bias_leakyrelu_op_impl, CUDA,
                     fused_bias_leakyrelu_op);

void GatherPointsForwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           const Tensor points,
                                           const Tensor idx, Tensor out);

void GatherPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                            const Tensor grad_out,
                                            const Tensor idx,
                                            Tensor grad_points);

void gather_points_forward_cuda(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out) {
  GatherPointsForwardCUDAKernelLauncher(b, c, n, npoints, points, idx, out);
};

void gather_points_backward_cuda(int b, int c, int n, int npoints,
                                 const Tensor grad_out, const Tensor idx,
                                 Tensor grad_points) {
  GatherPointsBackwardCUDAKernelLauncher(b, c, n, npoints, grad_out, idx,
                                         grad_points);
};

void gather_points_forward_impl(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out);

void gather_points_backward_impl(int b, int c, int n, int npoints,
                                 const Tensor grad_out, const Tensor idx,
                                 Tensor grad_points);

REGISTER_DEVICE_IMPL(gather_points_forward_impl, CUDA,
                     gather_points_forward_cuda);
REGISTER_DEVICE_IMPL(gather_points_backward_impl, CUDA,
                     gather_points_backward_cuda);

void GroupPointsForwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                          int nsample, const Tensor points,
                                          const Tensor idx, Tensor out);

void GroupPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           int nsample, const Tensor grad_out,
                                           const Tensor idx,
                                           Tensor grad_points);

void group_points_forward_cuda(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out) {
  GroupPointsForwardCUDAKernelLauncher(b, c, n, npoints, nsample, points, idx,
                                       out);
};

void group_points_backward_cuda(int b, int c, int n, int npoints, int nsample,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points) {
  GroupPointsBackwardCUDAKernelLauncher(b, c, n, npoints, nsample, grad_out,
                                        idx, grad_points);
};

void group_points_forward_impl(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out);

void group_points_backward_impl(int b, int c, int n, int npoints, int nsample,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points);

REGISTER_DEVICE_IMPL(group_points_forward_impl, CUDA,
                     group_points_forward_cuda);
REGISTER_DEVICE_IMPL(group_points_backward_impl, CUDA,
                     group_points_backward_cuda);

void IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(const int num_a,
                                                   const Tensor boxes_a,
                                                   const int num_b,
                                                   const Tensor boxes_b,
                                                   Tensor ans_overlap);

void IoU3DBoxesIoUBevForwardCUDAKernelLauncher(const int num_a,
                                               const Tensor boxes_a,
                                               const int num_b,
                                               const Tensor boxes_b,
                                               Tensor ans_iou);

void IoU3DNMSForwardCUDAKernelLauncher(const Tensor boxes,
                                       unsigned long long* mask, int boxes_num,
                                       float nms_overlap_thresh);

void IoU3DNMSNormalForwardCUDAKernelLauncher(const Tensor boxes,
                                             unsigned long long* mask,
                                             int boxes_num,
                                             float nms_overlap_thresh);

void iou3d_boxes_overlap_bev_forward_cuda(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap) {
  IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(num_a, boxes_a, num_b, boxes_b,
                                                ans_overlap);
};

void iou3d_boxes_iou_bev_forward_cuda(const int num_a, const Tensor boxes_a,
                                      const int num_b, const Tensor boxes_b,
                                      Tensor ans_iou) {
  IoU3DBoxesIoUBevForwardCUDAKernelLauncher(num_a, boxes_a, num_b, boxes_b,
                                            ans_iou);
};

void iou3d_nms_forward_cuda(const Tensor boxes, unsigned long long* mask,
                            int boxes_num, float nms_overlap_thresh) {
  IoU3DNMSForwardCUDAKernelLauncher(boxes, mask, boxes_num, nms_overlap_thresh);
};

void iou3d_nms_normal_forward_cuda(const Tensor boxes, unsigned long long* mask,
                                   int boxes_num, float nms_overlap_thresh) {
  IoU3DNMSNormalForwardCUDAKernelLauncher(boxes, mask, boxes_num,
                                          nms_overlap_thresh);
};

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap);

void iou3d_boxes_iou_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                      const int num_b, const Tensor boxes_b,
                                      Tensor ans_iou);

void iou3d_nms_forward_impl(const Tensor boxes, unsigned long long* mask,
                            int boxes_num, float nms_overlap_thresh);

void iou3d_nms_normal_forward_impl(const Tensor boxes, unsigned long long* mask,
                                   int boxes_num, float nms_overlap_thresh);

REGISTER_DEVICE_IMPL(iou3d_boxes_overlap_bev_forward_impl, CUDA,
                     iou3d_boxes_overlap_bev_forward_cuda);
REGISTER_DEVICE_IMPL(iou3d_boxes_iou_bev_forward_impl, CUDA,
                     iou3d_boxes_iou_bev_forward_cuda);
REGISTER_DEVICE_IMPL(iou3d_nms_forward_impl, CUDA, iou3d_nms_forward_cuda);
REGISTER_DEVICE_IMPL(iou3d_nms_normal_forward_impl, CUDA,
                     iou3d_nms_normal_forward_cuda);

void KNNForwardCUDAKernelLauncher(int b, int n, int m, int nsample,
                                  const Tensor xyz, const Tensor new_xyz,
                                  Tensor idx, Tensor dist2);

void knn_forward_cuda(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2) {
  KNNForwardCUDAKernelLauncher(b, n, m, nsample, xyz, new_xyz, idx, dist2);
}

void knn_forward_impl(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2);
REGISTER_DEVICE_IMPL(knn_forward_impl, CUDA, knn_forward_cuda);

void MaskedIm2colForwardCUDAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int kernel_h,
                                           const int kernel_w, const int pad_h,
                                           const int pad_w);

void MaskedCol2imForwardCUDAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int height,
                                           const int width, const int channels);

void masked_im2col_forward_cuda(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  MaskedIm2colForwardCUDAKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                        kernel_h, kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_cuda(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  MaskedCol2imForwardCUDAKernelLauncher(col, mask_h_idx, mask_w_idx, im, height,
                                        width, channels);
}

void masked_im2col_forward_impl(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w);

void masked_col2im_forward_impl(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels);

REGISTER_DEVICE_IMPL(masked_im2col_forward_impl, CUDA,
                     masked_im2col_forward_cuda);
REGISTER_DEVICE_IMPL(masked_col2im_forward_impl, CUDA,
                     masked_col2im_forward_cuda);

void modulated_deformable_im2col_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_deterministic_cuda(
    const Tensor data_offset, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int kh_id, const int kw_id,const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int g_id, Tensor add_index, Tensor add_value);
void modulated_deform_conv_fast_backward_cuda(Tensor grad, Tensor offset, Tensor input, Tensor  wh_mat, Tensor  offset_temp,
                                               const int channels_per_group,  const int height, const int width);
void modulated_deform_conv_fast_forward_cuda(Tensor offset, Tensor input, Tensor output,
                                               const int kernel_h, const int kernel_w, const int batch_size, const int deform_groups,
                                               const int channels_per_group,  const int height, const int width);
void grid_sample_deterministic_forward_cuda(Tensor  input,Tensor  offset,Tensor  output,
                                      const int offset_num, const int channel,  const int height, const int width);
void grid_sample_deterministic_input_backward_cuda(Tensor grad_output, Tensor t2_indice, Tensor offset, Tensor batch_id,
  Tensor t2_i_i_s_id, Tensor t2_i_i_f_id, Tensor grad_input, const int group,
  const int offset_num, const int channel, const int height, const int width);
void grid_sample_deterministic_offset_backward_cuda(Tensor grad_output, Tensor input, Tensor offset, Tensor grad_offset,
  const int offset_num, const int channel,  const int height, const int width);
void modulated_deformable_col2im_coord_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_im2col_impl(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_impl(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);
void modulated_deformable_col2im_deterministic_impl(
    const Tensor data_offset, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int kh_id, const int kw_id,const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int g_id, Tensor add_index, Tensor add_value);
void modulated_deformable_col2im_coord_impl(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);
void modulated_deform_conv_fast_backward_impl(Tensor grad, Tensor offset, Tensor input, Tensor  wh_mat, Tensor  offset_temp,
                                               const int channels_per_group,  const int height, const int width);
void modulated_deform_conv_fast_forward_impl(Tensor offset, Tensor input, Tensor output,
                                               const int kernel_h, const int kernel_w, const int batch_size, const int deform_groups,
                                               const int channels_per_group,  const int height, const int width);
void grid_sample_deterministic_forward_impl(Tensor  input,Tensor  offset,Tensor  output,
                                      const int offset_num, const int channel,  const int height, const int width);
void grid_sample_deterministic_input_backward_impl(Tensor grad_output, Tensor t2_indice, Tensor offset, Tensor batch_id,
  Tensor t2_i_i_s_id, Tensor t2_i_i_f_id, Tensor grad_input, const int group,
  const int offset_num, const int channel, const int height, const int width);
void grid_sample_deterministic_offset_backward_impl(Tensor grad_output, Tensor input, Tensor offset, Tensor grad_offset,
  const int offset_num, const int channel,  const int height, const int width);
REGISTER_DEVICE_IMPL(modulated_deformable_im2col_impl, CUDA,
                     modulated_deformable_im2col_cuda);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_impl, CUDA,
                     modulated_deformable_col2im_cuda);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_coord_impl, CUDA,
                     modulated_deformable_col2im_coord_cuda);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_deterministic_impl, CUDA,
                     modulated_deformable_col2im_deterministic_cuda);
REGISTER_DEVICE_IMPL(modulated_deform_conv_fast_forward_impl, CUDA,
                     modulated_deform_conv_fast_forward_cuda);
REGISTER_DEVICE_IMPL(modulated_deform_conv_fast_backward_impl, CUDA,
                     modulated_deform_conv_fast_backward_cuda);
REGISTER_DEVICE_IMPL(grid_sample_deterministic_forward_impl, CUDA,
                     grid_sample_deterministic_forward_cuda);
REGISTER_DEVICE_IMPL(grid_sample_deterministic_input_backward_impl, CUDA,
                     grid_sample_deterministic_input_backward_cuda);
REGISTER_DEVICE_IMPL(grid_sample_deterministic_offset_backward_impl, CUDA,
                     grid_sample_deterministic_offset_backward_cuda);
Tensor ms_deform_attn_cuda_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

void ms_deform_attn_cuda_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

Tensor ms_deform_attn_impl_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

void ms_deform_attn_impl_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_forward, CUDA,
                     ms_deform_attn_cuda_forward);
REGISTER_DEVICE_IMPL(ms_deform_attn_impl_backward, CUDA,
                     ms_deform_attn_cuda_backward);

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset);

Tensor nms_cuda(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSCUDAKernelLauncher(boxes, scores, iou_threshold, offset);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, CUDA, nms_cuda);

void PointsInBoxesPartForwardCUDAKernelLauncher(int batch_size, int boxes_num,
                                                int pts_num, const Tensor boxes,
                                                const Tensor pts,
                                                Tensor box_idx_of_points);

void PointsInBoxesAllForwardCUDAKernelLauncher(int batch_size, int boxes_num,
                                               int pts_num, const Tensor boxes,
                                               const Tensor pts,
                                               Tensor box_idx_of_points);

void points_in_boxes_part_forward_cuda(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points) {
  PointsInBoxesPartForwardCUDAKernelLauncher(batch_size, boxes_num, pts_num,
                                             boxes, pts, box_idx_of_points);
};

void points_in_boxes_all_forward_cuda(int batch_size, int boxes_num,
                                      int pts_num, const Tensor boxes,
                                      const Tensor pts,
                                      Tensor box_idx_of_points) {
  PointsInBoxesAllForwardCUDAKernelLauncher(batch_size, boxes_num, pts_num,
                                            boxes, pts, box_idx_of_points);
};

void points_in_boxes_part_forward_impl(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points);

void points_in_boxes_all_forward_impl(int batch_size, int boxes_num,
                                      int pts_num, const Tensor boxes,
                                      const Tensor pts,
                                      Tensor box_idx_of_points);
REGISTER_DEVICE_IMPL(points_in_boxes_part_forward_impl, CUDA,
                     points_in_boxes_part_forward_cuda);
REGISTER_DEVICE_IMPL(points_in_boxes_all_forward_impl, CUDA,
                     points_in_boxes_all_forward_cuda);

void PSAMaskForwardCUDAKernelLauncher(const int psa_type, const Tensor input,
                                      Tensor output, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask);

void PSAMaskBackwardCUDAKernelLauncher(
    const int psa_type, const Tensor grad_output, Tensor grad_input,
    const int num_, const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int half_h_mask, const int half_w_mask);

void psamask_forward_cuda(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask) {
  PSAMaskForwardCUDAKernelLauncher(psa_type, input, output, num_, h_feature,
                                   w_feature, h_mask, w_mask, half_h_mask,
                                   half_w_mask);
}

void psamask_backward_cuda(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask) {
  PSAMaskBackwardCUDAKernelLauncher(psa_type, grad_output, grad_input, num_,
                                    h_feature, w_feature, h_mask, w_mask,
                                    half_h_mask, half_w_mask);
}

void psamask_forward_impl(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask);

void psamask_backward_impl(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);
REGISTER_DEVICE_IMPL(psamask_forward_impl, CUDA, psamask_forward_cuda);
REGISTER_DEVICE_IMPL(psamask_backward_impl, CUDA, psamask_backward_cuda);

void ROIAlignForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       Tensor argmax_y, Tensor argmax_x,
                                       int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio,
                                       int pool_mode, bool aligned);

void ROIAlignBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor argmax_y, Tensor argmax_x,
                                        Tensor grad_input, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio, int pool_mode,
                                        bool aligned);
void ROIAlignDeterministicBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor argmax_y, Tensor argmax_x,
                                        Tensor index_add, Tensor add, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio, int pool_mode,
                                        bool aligned,int batch_size, int channels, int height, int width, Tensor sparse_index);
void roi_align_forward_cuda(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignForwardCUDAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned) {
  ROIAlignBackwardCUDAKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}
void roi_align_deterministic_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor index_add, Tensor add,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned,int batch_size, int channels, int height, int width, Tensor sparse_index) {
  ROIAlignDeterministicBackwardCUDAKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, index_add, add, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned, batch_size, channels, height, width, sparse_index);
}
void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned);
void roi_align_deterministic_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor index_add, Tensor add,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned, int batch_size, int channels, int height, int width, Tensor sparse_index);
REGISTER_DEVICE_IMPL(roi_align_forward_impl, CUDA, roi_align_forward_cuda);
REGISTER_DEVICE_IMPL(roi_align_backward_impl, CUDA, roi_align_backward_cuda);
REGISTER_DEVICE_IMPL(roi_align_deterministic_backward_impl, CUDA, roi_align_deterministic_backward_cuda);


void ROIAlignRotatedForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int sample_num, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor output);

void ROIAlignRotatedBackwardCUDAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int sample_num, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor bottom_grad);

void roi_align_rotated_forward_cuda(Tensor features, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sample_ratio,
                                    bool aligned, bool clockwise) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);
  ROIAlignRotatedForwardCUDAKernelLauncher(
      features, rois, spatial_scale, sample_ratio, aligned, clockwise,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width, output);
}

void roi_align_rotated_backward_cuda(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sample_ratio, bool aligned,
                                     bool clockwise) {
  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 6) {
    AT_ERROR("wrong roi size");
  }

  int num_channels = bottom_grad.size(1);
  int data_height = bottom_grad.size(2);
  int data_width = bottom_grad.size(3);
  ROIAlignRotatedBackwardCUDAKernelLauncher(
      top_grad, rois, spatial_scale, sample_ratio, aligned, clockwise,
      num_channels, data_height, data_width, num_rois, aligned_height,
      aligned_width, bottom_grad);
}

void roi_align_rotated_forward_impl(Tensor features, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sample_ratio,
                                    bool aligned, bool clockwise);

void roi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sample_ratio, bool aligned,
                                     bool clockwise);
REGISTER_DEVICE_IMPL(roi_align_rotated_forward_impl, CUDA,
                     roi_align_rotated_forward_cuda);
REGISTER_DEVICE_IMPL(roi_align_rotated_backward_impl, CUDA,
                     roi_align_rotated_backward_cuda);

void RoiawarePool3dForwardCUDAKernelLauncher(
    int boxes_num, int pts_num, int channels, int max_pts_each_voxel, int out_x,
    int out_y, int out_z, const Tensor rois, const Tensor pts,
    const Tensor pts_feature, Tensor argmax, Tensor pts_idx_of_voxels,
    Tensor pooled_features, int pool_method);

void RoiawarePool3dBackwardCUDAKernelLauncher(
    int boxes_num, int out_x, int out_y, int out_z, int channels,
    int max_pts_each_voxel, const Tensor pts_idx_of_voxels, const Tensor argmax,
    const Tensor grad_out, Tensor grad_in, int pool_method);

void roiaware_pool3d_forward_cuda(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method) {
  RoiawarePool3dForwardCUDAKernelLauncher(
      boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
      rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features,
      pool_method);
};

void roiaware_pool3d_backward_cuda(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method) {
  RoiawarePool3dBackwardCUDAKernelLauncher(
      boxes_num, out_x, out_y, out_z, channels, max_pts_each_voxel,
      pts_idx_of_voxels, argmax, grad_out, grad_in, pool_method);
};

void roiaware_pool3d_forward_impl(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method);

void roiaware_pool3d_backward_impl(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method);

REGISTER_DEVICE_IMPL(roiaware_pool3d_forward_impl, CUDA,
                     roiaware_pool3d_forward_cuda);
REGISTER_DEVICE_IMPL(roiaware_pool3d_backward_impl, CUDA,
                     roiaware_pool3d_backward_cuda);

void RoIPointPool3dForwardCUDAKernelLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len,
    int sampled_pts_num, const Tensor xyz, const Tensor boxes3d,
    const Tensor pts_feature, Tensor pooled_features, Tensor pooled_empty_flag);

void roipoint_pool3d_forward_cuda(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag) {
  RoIPointPool3dForwardCUDAKernelLauncher(
      batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, xyz,
      boxes3d, pts_feature, pooled_features, pooled_empty_flag);
};

void roipoint_pool3d_forward_impl(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag);
REGISTER_DEVICE_IMPL(roipoint_pool3d_forward_impl, CUDA,
                     roipoint_pool3d_forward_cuda);

void ROIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax, int pooled_height,
                                      int pooled_width, float spatial_scale);

void ROIPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                       Tensor argmax, Tensor grad_input,
                                       int pooled_height, int pooled_width,
                                       float spatial_scale);

void roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale) {
  ROIPoolForwardCUDAKernelLauncher(input, rois, output, argmax, pooled_height,
                                   pooled_width, spatial_scale);
}

void roi_pool_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale) {
  ROIPoolBackwardCUDAKernelLauncher(grad_output, rois, argmax, grad_input,
                                    pooled_height, pooled_width, spatial_scale);
}

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);
void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);
REGISTER_DEVICE_IMPL(roi_pool_forward_impl, CUDA, roi_pool_forward_cuda);
REGISTER_DEVICE_IMPL(roi_pool_backward_impl, CUDA, roi_pool_backward_cuda);

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

std::vector<at::Tensor> DynamicPointToVoxelForwardCUDAKernelLauncher(
    const at::Tensor& feats, const at::Tensor& coors,
    const reduce_t reduce_type);

void DynamicPointToVoxelBackwardCUDAKernelLauncher(
    at::Tensor& grad_feats, const at::Tensor& grad_reduced_feats,
    const at::Tensor& feats, const at::Tensor& reduced_feats,
    const at::Tensor& coors_map, const at::Tensor& reduce_count,
    const reduce_t reduce_type);

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_cuda(
    const torch::Tensor& feats, const torch::Tensor& coors,
    const reduce_t reduce_type) {
  return DynamicPointToVoxelForwardCUDAKernelLauncher(feats, coors,
                                                      reduce_type);
};

void dynamic_point_to_voxel_backward_cuda(
    torch::Tensor& grad_feats, const torch::Tensor& grad_reduced_feats,
    const torch::Tensor& feats, const torch::Tensor& reduced_feats,
    const torch::Tensor& coors_idx, const torch::Tensor& reduce_count,
    const reduce_t reduce_type) {
  DynamicPointToVoxelBackwardCUDAKernelLauncher(grad_feats, grad_reduced_feats,
                                                feats, reduced_feats, coors_idx,
                                                reduce_count, reduce_type);
};

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_impl(
    const torch::Tensor& feats, const torch::Tensor& coors,
    const reduce_t reduce_type);

void dynamic_point_to_voxel_backward_impl(
    torch::Tensor& grad_feats, const torch::Tensor& grad_reduced_feats,
    const torch::Tensor& feats, const torch::Tensor& reduced_feats,
    const torch::Tensor& coors_idx, const torch::Tensor& reduce_count,
    const reduce_t reduce_type);

REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_forward_impl, CUDA,
                     dynamic_point_to_voxel_forward_cuda);
REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_backward_impl, CUDA,
                     dynamic_point_to_voxel_backward_cuda);

void SyncBNForwardMeanCUDAKernelLauncher(const Tensor input, Tensor mean);

void SyncBNForwardVarCUDAKernelLauncher(const Tensor input, const Tensor mean,
                                        Tensor var);

void SyncBNForwardOutputCUDAKernelLauncher(
    const Tensor input, const Tensor mean, const Tensor var,
    Tensor running_mean, Tensor running_var, const Tensor weight,
    const Tensor bias, Tensor norm, Tensor std, Tensor output, float eps,
    float momentum, int group_size);

void SyncBNBackwardParamCUDAKernelLauncher(const Tensor grad_output,
                                           const Tensor norm,
                                           Tensor grad_weight,
                                           Tensor grad_bias);

void SyncBNBackwardDataCUDAKernelLauncher(const Tensor grad_output,
                                          const Tensor weight,
                                          const Tensor grad_weight,
                                          const Tensor grad_bias,
                                          const Tensor norm, const Tensor std,
                                          Tensor grad_input);

void sync_bn_forward_mean_cuda(const Tensor input, Tensor mean) {
  SyncBNForwardMeanCUDAKernelLauncher(input, mean);
}

void sync_bn_forward_var_cuda(const Tensor input, const Tensor mean,
                              Tensor var) {
  SyncBNForwardVarCUDAKernelLauncher(input, mean, var);
}

void sync_bn_forward_output_cuda(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size) {
  SyncBNForwardOutputCUDAKernelLauncher(input, mean, var, running_mean,
                                        running_var, weight, bias, norm, std,
                                        output, eps, momentum, group_size);
}

void sync_bn_backward_param_cuda(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias) {
  SyncBNBackwardParamCUDAKernelLauncher(grad_output, norm, grad_weight,
                                        grad_bias);
}

void sync_bn_backward_data_cuda(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input) {
  SyncBNBackwardDataCUDAKernelLauncher(grad_output, weight, grad_weight,
                                       grad_bias, norm, std, grad_input);
}

void sync_bn_forward_mean_impl(const Tensor input, Tensor mean);

void sync_bn_forward_var_impl(const Tensor input, const Tensor mean,
                              Tensor var);

void sync_bn_forward_output_impl(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size);

void sync_bn_backward_param_impl(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data_impl(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input);

REGISTER_DEVICE_IMPL(sync_bn_forward_mean_impl, CUDA,
                     sync_bn_forward_mean_cuda);
REGISTER_DEVICE_IMPL(sync_bn_forward_var_impl, CUDA, sync_bn_forward_var_cuda);
REGISTER_DEVICE_IMPL(sync_bn_forward_output_impl, CUDA,
                     sync_bn_forward_output_cuda);
REGISTER_DEVICE_IMPL(sync_bn_backward_param_impl, CUDA,
                     sync_bn_backward_param_cuda);
REGISTER_DEVICE_IMPL(sync_bn_backward_data_impl, CUDA,
                     sync_bn_backward_data_cuda);

void ThreeInterpolateForwardCUDAKernelLauncher(int b, int c, int m, int n,
                                               const Tensor points,
                                               const Tensor idx,
                                               const Tensor weight, Tensor out);

void ThreeInterpolateBackwardCUDAKernelLauncher(int b, int c, int n, int m,
                                                const Tensor grad_out,
                                                const Tensor idx,
                                                const Tensor weight,
                                                Tensor grad_points);

void three_interpolate_forward_cuda(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out) {
  ThreeInterpolateForwardCUDAKernelLauncher(b, c, m, n, points, idx, weight,
                                            out);
};

void three_interpolate_backward_cuda(int b, int c, int n, int m,
                                     const Tensor grad_out, const Tensor idx,
                                     const Tensor weight, Tensor grad_points) {
  ThreeInterpolateBackwardCUDAKernelLauncher(b, c, n, m, grad_out, idx, weight,
                                             grad_points);
};

void three_interpolate_forward_impl(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out);

void three_interpolate_backward_impl(int b, int c, int n, int m,
                                     const Tensor grad_out, const Tensor idx,
                                     const Tensor weight, Tensor grad_points);
REGISTER_DEVICE_IMPL(three_interpolate_forward_impl, CUDA,
                     three_interpolate_forward_cuda);
REGISTER_DEVICE_IMPL(three_interpolate_backward_impl, CUDA,
                     three_interpolate_backward_cuda);

void ThreeNNForwardCUDAKernelLauncher(int b, int n, int m, const Tensor unknown,
                                      const Tensor known, Tensor dist2,
                                      Tensor idx);

void three_nn_forward_cuda(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx) {
  ThreeNNForwardCUDAKernelLauncher(b, n, m, unknown, known, dist2, idx);
};

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx);
REGISTER_DEVICE_IMPL(three_nn_forward_impl, CUDA, three_nn_forward_cuda);

void TINShiftForwardCUDAKernelLauncher(Tensor input, Tensor shift,
                                       Tensor output);

void TINShiftBackwardCUDAKernelLauncher(Tensor grad_output, Tensor shift,
                                        Tensor grad_input);

void tin_shift_forward_cuda(Tensor input, Tensor shift, Tensor output) {
  TINShiftForwardCUDAKernelLauncher(input, shift, output);
}

void tin_shift_backward_cuda(Tensor grad_output, Tensor shift,
                             Tensor grad_input) {
  TINShiftBackwardCUDAKernelLauncher(grad_output, shift, grad_input);
}

void tin_shift_forward_impl(Tensor input, Tensor shift, Tensor output);
void tin_shift_backward_impl(Tensor grad_output, Tensor shift,
                             Tensor grad_input);
REGISTER_DEVICE_IMPL(tin_shift_forward_impl, CUDA, tin_shift_forward_cuda);
REGISTER_DEVICE_IMPL(tin_shift_backward_impl, CUDA, tin_shift_backward_cuda);

torch::Tensor upfirdn2d_op(const torch::Tensor& input,
                           const torch::Tensor& kernel, int up_x, int up_y,
                           int down_x, int down_y, int pad_x0, int pad_x1,
                           int pad_y0, int pad_y1);

torch::Tensor upfirdn2d_op_impl(const torch::Tensor& input,
                                const torch::Tensor& kernel, int up_x, int up_y,
                                int down_x, int down_y, int pad_x0, int pad_x1,
                                int pad_y0, int pad_y1);
REGISTER_DEVICE_IMPL(upfirdn2d_op_impl, CUDA, upfirdn2d_op);

int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor& points, at::Tensor& voxels, at::Tensor& coors,
    at::Tensor& num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor& points, at::Tensor& coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3);

int hard_voxelize_forward_cuda(const at::Tensor& points, at::Tensor& voxels,
                               at::Tensor& coors,
                               at::Tensor& num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim) {
  return HardVoxelizeForwardCUDAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

void dynamic_voxelize_forward_cuda(const at::Tensor& points, at::Tensor& coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim) {
  DynamicVoxelizeForwardCUDAKernelLauncher(points, coors, voxel_size,
                                           coors_range, NDim);
};

int hard_voxelize_forward_impl(const at::Tensor& points, at::Tensor& voxels,
                               at::Tensor& coors,
                               at::Tensor& num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim);

void dynamic_voxelize_forward_impl(const at::Tensor& points, at::Tensor& coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim);
REGISTER_DEVICE_IMPL(hard_voxelize_forward_impl, CUDA,
                     hard_voxelize_forward_cuda);
REGISTER_DEVICE_IMPL(dynamic_voxelize_forward_impl, CUDA,
                     dynamic_voxelize_forward_cuda);
