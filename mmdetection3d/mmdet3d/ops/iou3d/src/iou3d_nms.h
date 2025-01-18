#ifndef IOU3D_NMS_H
#define IOU3D_NMS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap, int device_id);
int boxes_overlap_3d_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap, int device_id);
int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou, int device_id);
int boxes_iou_3d_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou, int device_id);
int boxes_iou_3d_aligned_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou, int device_id);
int nms_bev_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id);
int nms_3d_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id);

#endif
