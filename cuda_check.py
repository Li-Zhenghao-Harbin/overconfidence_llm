import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"PyTorch CUDA版本: {torch.version.cuda}")
print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 运行一个真正的计算任务，这是关键步骤
try:
    x = torch.randn(10).cuda()
    print(f"在GPU上创建随机张量成功: {x}")
    print("RTX 5060 GPU 已成功被 PyTorch 驱动！")
except Exception as e:
    print(f"GPU 测试失败，遇到了错误: {e}")
    print("这可能意味着某些特定操作尚未被 Nightly 版本支持。")