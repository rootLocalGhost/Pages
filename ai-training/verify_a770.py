import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"XPU Available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Device Name: {torch.xpu.get_device_name(0)}")
    print(f"Memory Available: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Run test matrix multiplication using bfloat16 on XMX
    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    y = torch.matmul(x, x)
    print(f"Tensor Compute Test: SUCCESS ({y.shape})")