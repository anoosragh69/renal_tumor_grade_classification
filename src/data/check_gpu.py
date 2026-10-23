import torch, sys, platform

def main():
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split(' ')[0]}")
    print(f"Torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version (torch): {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device detected.")

if __name__ == "__main__":
    main()
