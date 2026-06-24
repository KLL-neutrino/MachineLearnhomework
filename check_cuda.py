import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

    x = torch.randn(3, 3).cuda()
    print("Tensor device:", x.device)
    print(x)
else:
    print("CUDA 不可用，请检查 PyTorch 是否为 CUDA 版本。")
