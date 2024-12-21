import torch


def main():
    tensor = torch.randn(100000000).double()
    tensors = []

    while True:
        cloned = tensor.clone().to("cuda")
        tensors.append(cloned)
        print(f"Allocated tensor on {cloned.device}, total tensors: {len(tensors)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("CUDA is not available.")
