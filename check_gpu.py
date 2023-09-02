import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available.")

print()

if torch.backends.mps.is_available():
    print("Metal/MPS is available!")
else:
    print("Metal/MPS is not available.")

print("If you want to check the version of Metal and MPS on your macOS device, you can go to \"About This Mac\" -> \"System Report\" -> \"Graphics/Displays\" and look for information related to Metal and MPS.")

print()

if torch.version.hip is not None:
    print("ROCm is available!")
    print("ROCm version:", torch.version.hip)
else:
    print("ROCm is not available.")
