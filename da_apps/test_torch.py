# da_apps/test_torch.py
import torch

print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("torch cuda version:", torch.version.cuda)
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        resv  = torch.cuda.memory_reserved() / (1024**3)
        print(f"{tag} {alloc:.3f} GB allocated {resv:.3f} GB reserved")

mem("before")

# ACTUAL CUDA allocation (about ~0.5 GB)
x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
mem("after alloc")

# Do something to ensure it isn't optimized away (not really needed, but ok)
y = (x @ x.T)
mem("after matmul")

del x, y
torch.cuda.empty_cache()
mem("after free")
