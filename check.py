import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Training on: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("❌ GPU not detected. Using CPU.")

# Simple test: Move a tensor to GPU
x = torch.rand(5, 3).to(device)
print(x)