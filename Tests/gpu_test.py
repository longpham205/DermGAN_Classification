import torch

x = torch.randn(6000, 6000, device="cuda")
y = torch.matmul(x, x)

print("GPU computation OK")
