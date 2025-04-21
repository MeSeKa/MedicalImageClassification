import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Bu True olmalı

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch built with): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(
        f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print(">>> CUDA DESTEĞİ BULUNAMADI! PyTorch kurulumunu kontrol et.")
