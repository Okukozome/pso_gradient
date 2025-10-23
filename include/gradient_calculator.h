#ifndef __GRADIENT_CALCULATOR_H
#define __GRADIENT_CALCULATOR_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <utility> // for std::pair

/**
 * @brief 结构体，用于存放所有（先前在 iPEPS.cpp 中硬编码的）物理模型参数
 *
 */
struct ModelParameters {
    double J1 = 1.0;
    double J2 = 0.57;
    double delta = 0.1;
    double deltaX = 0.0;
    double deltaY = 0.0;
};

/**
 * @brief 核心梯度计算函数
 * * @param ipeps_tensors_in 输入的 iPEPS 张量 (即 "x")。应为 kF64 (Double) 类型。
 * @param model_params 包含 J1, J2, delta 等物理参数的结构体。
 * @param fidx 文件索引，用于加载和保存 CTMRG 环境。
 * @param chi CTMRG 的截断维数 (KAI)。
 * @param epspr CTMRG 迭代的收敛精度。
 * @param ctm_state CTMRG 的初始状态 ('C' = 从文件继续, 'I' = 随机初始化)。
 * @return std::pair<torch::Tensor, std::vector<torch::Tensor>> 
 * 返回一个包含 (能量, 梯度列表) 的 std::pair。
 */
std::pair<torch::Tensor, std::vector<torch::Tensor>> get_energy_and_gradient(
    const std::vector<torch::Tensor>& ipeps_tensors_in,
    const ModelParameters& model_params,
    int fidx,
    int chi,
    std::pair<double, double> epspr,
    char ctm_state = 'C'
);

#endif // __GRADIENT_CALCULATOR_H