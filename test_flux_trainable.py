import torch
from models.flux_i2i_trainable import FluxI2ITrainable
import gc

def print_memory_usage(step_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# 1. 清理之前的垃圾显存 (好习惯)
torch.cuda.empty_cache()
gc.collect()

print("⏳ Initializing FLUX 4-bit Model on 5070 Ti...")

# 2. 实例化模型 (这里会自动触发 bitsandbytes 量化加载)
model = FluxI2ITrainable().cuda()

print_memory_usage("Model Loaded")

# 3. 创建假数据
# 注意：FLUX 默认训练是 1024x1024，但为了测试显存，我们先用 512x512
print("Creating dummy input (1, 3, 512, 512)...")
# 加上类型转换，让输入数据变成 FP16
x = torch.randn(1, 3, 512, 512).cuda().to(dtype=torch.float16).clamp(-1, 1)

# 4. 前向传播 (Forward Pass)
print("Running Forward Pass (this might take a few seconds)...")

# 注意：我们暂时用 no_grad 跑一遍 sanity check
# 因为 4-bit 模型默认是不支持直接 backward 的，需要加 LoRA 之后才行
# 这里只要能跑通 forward，说明环境和模型加载就成功了
with torch.no_grad():
    # 现在返回 (loss, image)
    loss, y = model(x, num_steps=1000) 

# 5. 验证输出
print("-" * 30)
print(f"Sample Loss: {loss.item():.4f}")
print(f"Output Shape: {y.shape}")
# ... 其余不变
# 5. 验证输出
print("-" * 30)
print(f"Output Shape: {y.shape}")
print(f"Min value:   {y.min().item():.3f}")
print(f"Max value:   {y.max().item():.3f}")
print("-" * 30)

if not torch.isnan(y).any():
    print("Test Passed! Model is loaded and runnable.")
    print("下一步：我们可以准备加上 LoRA 进行真正的训练了。")
else:
    print("Error: Output contains NaN!")