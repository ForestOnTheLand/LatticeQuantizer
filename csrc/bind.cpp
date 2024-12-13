#include "core.h"

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


void clip_wrapped(
    py::array_t<double, py::array::c_style | py::array::forcecast> mat,
    py::array_t<double, py::array::c_style | py::array::forcecast> vec,
    py::array_t<int, py::array::c_style | py::array::forcecast> result
) {
    // std::cout << mat.ref_count() << ", " << vec.ref_count() << ", " << result.ref_count() << std::endl;
    py::buffer_info mat_info = mat.request(), vec_info = vec.request(), result_info = result.request();
    clip(
        mat_info.shape[0],
        static_cast<double*>(mat_info.ptr),
        static_cast<double*>(vec_info.ptr),
        static_cast<int*>(result_info.ptr)
    );
}

void reduce_wrapped(py::array_t<double, py::array::c_style | py::array::forcecast> in, double delta) {
    py::buffer_info info = in.request();
    reduce(info.shape[0], static_cast<double*>(info.ptr), delta);
}

PYBIND11_MODULE(csrc, m) {
    m.def("reduce", &reduce_wrapped, "Reduction function, based on Lenstra–Lenstra–Lovasz algorithm.");
    m.def("clip", &clip_wrapped, "Find the closest lattice point to a given point.");
}
