#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include "roi_align_kernel.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("roi_align_forward_cuda", &roi_align_forward_cuda, "roi_align_forward_cuda",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));

    m.def("roi_align_backward_cuda", &roi_align_backward_cuda, "roi_align_backward_cuda",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
        py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
}