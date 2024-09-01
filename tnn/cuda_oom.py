import torch


def main():
    tensor = torch.randn(10000).double()
    tensors = []

    while True:
        cloned = tensor.clone().to("cuda")
        tensors.append(cloned)


if __name__ == "__main__":
    main()
