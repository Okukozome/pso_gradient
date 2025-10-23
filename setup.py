from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 从 localconfig.h 读取宏定义，以便 C++ 代码可以访问
#
# 这是一个好的实践，但为了简单起见，我们也可以
# 假设 localconfig.h 只是被 C++ #include
# 我们将保持简单，只列出源文件。
# C++ 代码将通过 #include "localconfig.h" 正常工作。

setup(
    name='ipeps_torch_extension', # Python 模块名
    ext_modules=[
        CppExtension(
            'ipeps_torch_extension', # 模块名 (必须与 PYBIND11_MODULE 匹配)
            [
                'src/pybind_wrapper.cpp',
                'src/gradient_calculator.cpp',
                'src/iPEPS.cpp' # 包含 iPEPS 基类实现
            ],
            include_dirs=[
                'include/' # 告诉编译器在哪里查找 .h 文件
            ],
            extra_compile_args=[
                '-std=c++17',      # 匹配您的原始 makefile
                '-O2',             # 优化
                '-fopenmp',        # 启用 OpenMP
                '-D_GLIBCXX_USE_CXX11_ABI=1' # 匹配您的 makefile
                # '-ggdb' # 如果需要调试，请取消注释
            ],
            extra_link_args=[
                '-fopenmp'
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)