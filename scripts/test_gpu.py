import os
os.environ["CUDA_FORCE_PTX_JIT"] = "1"
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    try:
        x = torch.randn(100, 100, device="cuda")
        y = x @ x
        print("CUDA Tensor MatMul Success! Result shape:", y.shape)
    except Exception as e:
        print("CUDA Error during matmul:", e)
else:
    print("CUDA not available.")
