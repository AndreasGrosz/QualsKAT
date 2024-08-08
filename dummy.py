import torch
import psutil

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB / {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

    print(f"CPU Memory Usage: {psutil.virtual_memory().percent}%")

# Verwenden Sie diese Funktion an strategischen Stellen in Ihrem Code, z.B.:
# print_gpu_memory_usage()
