#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "iou3d_cpu.h"
#include "iou3d_nms.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes bev overlap");
	m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes bev iou");
	m.def("boxes_overlap_3d_gpu", &boxes_overlap_3d_gpu, "oriented boxes 3d overlap");
	m.def("boxes_iou_3d_gpu", &boxes_iou_3d_gpu, "oriented boxes 3d iou");
	m.def("boxes_iou_3d_aligned_gpu", &boxes_iou_3d_aligned_gpu, "oriented boxes 3d iou aligned");
	m.def("nms_3d_gpu", &nms_3d_gpu, "nms 3d gpu");
	m.def("nms_bev_gpu", &nms_bev_gpu, "nms bev gpu");
	m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");
}
