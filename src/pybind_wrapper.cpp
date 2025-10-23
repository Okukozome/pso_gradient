#include <torch/extension.h> // 关键头文件，包含 pybind11 和 tensor 转换
#include <pybind11/stl.h>     // 用于 C++ STL (vector, pair)
#include "gradient_calculator.h" // 我们的 C++ 接口

namespace py = pybind11;

// PYBIND11_MODULE 定义了 Python 模块
// "ipeps_torch_extension" 是 `import` 时使用的模块名
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "iPEPS gradient calculator C++/PyTorch extension";

    // 1. 绑定 ModelParameters 结构体
    // 这会将其暴露为 Python 中的一个类
    py::class_<ModelParameters>(m, "ModelParameters")
        .def(py::init<>()) // 默认构造函数
        .def_readwrite("J1", &ModelParameters::J1)
        .def_readwrite("J2", &ModelParameters::J2)
        .def_readwrite("delta", &ModelParameters::delta)
        .def_readwrite("deltaX", &ModelParameters::deltaX)
        .def_readwrite("deltaY", &ModelParameters::deltaY)
        .def("__repr__", // 添加一个打印函数，方便调试
            [](const ModelParameters &p) {
                return "<ModelParameters: J1=" + std::to_string(p.J1) +
                       ", J2=" + std::to_string(p.J2) + ">";
            });

    // 2. 绑定核心函数 get_energy_and_gradient
    m.def("get_energy_and_gradient", 
          &get_energy_and_gradient, // C++ 函数指针
          "Calculates iPEPS ground state energy and gradient.", // docstring
          // --- 参数绑定 ---
          py::arg("ipeps_tensors_in"),
          py::arg("model_params"),
          py::arg("fidx"),
          py::arg("chi"),
          py::arg("epspr"),
          py::arg("ctm_state") = 'C', // 默认参数
          
          // --- 关键：释放 GIL ---
          // 告诉 pybind11，在调用这个 C++ 函数时，
          // 释放 Python 的全局解释器锁 (GIL)。
          // 这允许 Python 的其他线程（如下探）在 C++ 运行时继续工作。
          py::call_guard<py::gil_scoped_release>()
    );
}