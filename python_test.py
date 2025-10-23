import torch
import ipeps_torch_extension as ipeps_ext # 导入我们编译的 C++ 模块
import time
import os

# --- 1. 从 localconfig.h 同步配置 ---
# 这些值必须与 C++ 的 localconfig.h 匹配
DIMS = 2
DB = 2
LY = 2
LX = 2
CHI = (2 * DB * DB) # KAI

print(f"--- iPEPS Gradient Test ---")
print(f"Config: LX={LX}, LY={LY}, DB={DB}, CHI={CHI}")
print(f"PyTorch C++ extension: {ipeps_ext}")

# --- 2. 准备输入参数 ---

# a) 创建 iPEPS 张量 (x)
# 必须是 torch.float64 (double) 才能匹配 C++ 签名
# 必须匹配 (dims, DB, DB, DB, DB)
ipeps_tensors = []
for _ in range(LX * LY):
    # 使用 torch.randn 创建随机初始张量
    t = torch.randn((DIMS, DB, DB, DB, DB), dtype=torch.float64)
    ipeps_tensors.append(t)
    
print(f"\nCreated {len(ipeps_tensors)} input tensors.")
print(f"Tensor shape: {ipeps_tensors[0].shape}, Dtype: {ipeps_tensors[0].dtype}")

# b) 实例化 C++ 中的 ModelParameters 类
params = ipeps_ext.ModelParameters()
params.J1 = 1.0
params.J2 = 0.57 # 您可以修改这些值
print(f"Model Parameters: {params}")

# c) 其他参数
fidx = 0      # 文件索引 0
epspr = (1.0e-4, 1.0e-2) # 收敛精度
ctm_state = 'I' # 'I' = 随机初始化 (首次运行时更安全)

# 确保保存 CTM 文件的目录存在 (如果 fidx=0, 可能会创建 0_Clb_0 等)
# 您的 C++ 代码似乎将它们保存在当前工作目录中
print(f"\nUsing fidx={fidx}, chi={CHI}, ctm_state='{ctm_state}'")
print("Ensure CTM files (e.g., 0_Clb_0) can be written to this directory.")
# 首次运行时清理旧的 CTM 文件
try:
    for i in range(LX * LY):
        os.remove(f"{fidx}_Clb_{i}")
except FileNotFoundError:
    pass # 没关系


# --- 3. 调用 C++ 核心函数 ---
print("\nCalling C++ get_energy_and_gradient...")
print("(This will take time as it runs CTMRG)")
start_time = time.time()

# 这是神奇的地方：Python -> C++ -> Python
energy, gradients = ipeps_ext.get_energy_and_gradient(
    ipeps_tensors,
    params,
    fidx,
    CHI,
    epspr,
    ctm_state
)

end_time = time.time()
print(f"\n--- C++ Execution Finished (Time: {end_time - start_time:.4f}s) ---")

# --- 4. 检查输出 ---
print(f"\nReceived Energy (E):")
print(energy)
print(f"Energy per site: {energy.item() / (LX * LY):.10f}")

print(f"\nReceived {len(gradients)} Gradient Tensors:")
for i, grad in enumerate(gradients):
    print(f"  Grad[{i}]: shape={grad.shape}, dtype={grad.dtype}, norm={grad.norm().item():.4f}")

# --- 5. 验证 PyTorch 自动微分 (可选但推荐) ---
# 现在 ipeps_tensors 仍然是叶子节点，我们可以将它们包装在
# 一个 Python 函数中，并使用 PyTorch 的 Autograd 来检查
# (这需要 C++ 端正确实现了 autograd::Function，您的代码已经实现了)

# 注意：要使 C++ 的自定义 autograd 对 Python 可见，
# 您需要将 ipeps_tensors 设置为 requires_grad=True
print("\n--- Verifying Autograd Chain (Python side) ---")
for t in ipeps_tensors:
    t.requires_grad_(True)

# 再次调用，但这次 PyTorch 会构建图
ctm_state = 'C' # 'C' = 从上一步继续
print(f"Calling C++ again with ctm_state='{ctm_state}' and requires_grad=True...")
energy_torch, _ = ipeps_ext.get_energy_and_gradient(
    ipeps_tensors,
    params,
    fidx,
    CHI,
    epspr,
    ctm_state
)

print(f"\nEnergy from second call: {energy_torch.item() / (LX * LY):.10f}")

# 在 Python 端调用 backward()
try:
    energy_torch.backward()
    print("Python-side .backward() call SUCCESSFUL.")
    
    print("\nGradients computed by PyTorch:")
    for i, t in enumerate(ipeps_tensors):
        if t.grad is not None:
            print(f"  Tensor[{i}].grad: shape={t.grad.shape}, norm={t.grad.norm().item():.4f}")
        else:
            print(f"  Tensor[{i}].grad: None (Error!)")
            
except RuntimeError as e:
    print(f"\nPython-side .backward() call FAILED: {e}")
    print("This likely means the custom C++ autograd Function (StepCTMRG) ")
    print("is not correctly propagating gradients back to the initial inputs.")