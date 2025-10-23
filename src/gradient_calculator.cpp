#include "gradient_calculator.h"

// --- 包含所有必要的原始项目头文件 ---
// 确保 #include "..." 的路径相对于编译器设置
#include "iPEPS.h"
#include "TensorModel.h"
#include "ctmrg.h"
#include "StepCTMRG.h"
#include "CornerContraction.h"
#include "TensorExpectation.h"
#include "autograd.h"
#include "stringname.h"
#include "localconfig.h"
#include "globalconfig.h"

#include <stdexcept>
#include <iostream>

/**
 * @brief 一个 iPEPS 包装类，继承自原始的 iPEPS 类
 * 它的唯一目的是在构造时接收 ModelParameters，
 * 并用这些参数覆盖基类 iPEPS 中硬编码的物理参数。
 */
class GradientiPEPS : public iPEPS {
public:
    GradientiPEPS(std::vector<torch::Tensor> &gyx, const ModelParameters& params, bool ifgrad = true)
        : iPEPS(gyx, ifgrad) // 1. 调用基类构造函数
    {
        // 2. 基类构造函数会设置 J1=1.0, J2=0.57 等硬编码值
        // 3. 我们在这里立即用传入的参数覆盖它们
        this->J1 = params.J1;
        this->J2 = params.J2;
        this->delta = params.delta;
        this->deltaX = params.deltaX;
        this->deltaY = params.deltaY;
    }
};


/**
 * @brief 核心梯度计算函数的实现
 */
std::pair<torch::Tensor, std::vector<torch::Tensor>> get_energy_and_gradient(
    const std::vector<torch::Tensor>& ipeps_tensors_in,
    const ModelParameters& model_params,
    int fidx,
    int chi,
    std::pair<double, double> epspr,
    char ctm_state
) {
    std::vector<torch::Tensor> gradients;
    std::vector<torch::Tensor> wavefunc; 
    torch::Tensor local_energy;

    // 检查 localconfig.h 与输入张量列表大小是否匹配
    if (ipeps_tensors_in.size() != LY * LX) {
        throw std::runtime_error("Input tensor list size (" 
            + std::to_string(ipeps_tensors_in.size()) 
            + ") does not match localconfig.h LX*LY (" 
            + std::to_string(LX * LX) + ")");
    }

    try {
        // --- 1. 创建可追踪的张量副本 ---
        // 原始代码在 LBFGS 中使用 'islowered' 标志来决定是否转为 kFloat
        // groundEnergy 内部似乎期望 kFloat。我们将遵循此模式：
        
        wavefunc.reserve(ipeps_tensors_in.size());
        for (const auto& t : ipeps_tensors_in) {
            if (t.dtype() != torch::kF64) {
                 std::cerr << "Warning: Input tensor is not kF64. Converting." << std::endl;
            }
            // 转换为 kFloat，克隆，并设置 requires_grad_(true)
            wavefunc.push_back(t.to(torch::kFloat).clone().detach().requires_grad_(true));
        }

        // --- 2. 实例化使用自定义参数的 iPEPS 模型 ---
        GradientiPEPS sq(wavefunc, model_params, true);

        // --- 3. 前向传播 (计算能量 E) ---
        // 这对应于 optimizationLocal.h 中 closure 的第一部分
        // 注意：我们必须开启 ifdynamic=true 来运行 CTMRG，
        // 并且 iftodisk=true 来保存环境以供下一次迭代使用。
        local_energy = sq.groundEnergy(fidx, 
                                      ctm_state, 
                                      chi, 
                                      epspr, 
                                      true,  // ifdynamic
                                      false, // ifexpt
                                      true,  // iftodisk
                                      false  // printable
                                      );

        // --- 4. 反向传播 (计算梯度 dE/dx) ---
        // 这对应于 closure 中的 local_energy.backward()
        local_energy.backward();

        // --- 5. “提取”梯度 ---
        gradients.reserve(wavefunc.size());
        for (auto& t : wavefunc) {
            if (t.grad().defined()) {
                // 将 kFloat 类型的梯度转回 kF64 (Double) 以匹配输入
                gradients.push_back(t.grad().to(torch::kF64).clone());
            } else {
                std::cerr << "Warning: Tensor had no gradient defined." << std::endl;
                gradients.push_back(torch::zeros_like(t, torch::kF64));
            }
        }
        
        // 返回能量（转为 kF64）和梯度
        return {local_energy.to(torch::kF64).clone().detach(), gradients};

    } catch (const c10::Error& e) {
        std::cerr << "[C10 Exception in get_energy_and_gradient]: " << e.what() << std::endl;
        throw; // 重新抛出异常，让 pybind11 捕获并转换为 Python 异常
    } catch (const std::exception& e) {
        std::cerr << "[STD Exception in get_energy_and_gradient]: " << e.what() << std::endl;
        throw; // 重新抛出
    }
}