import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

from src.utils import train, test
from datasets import load_dataset
from matplotlib.animation import FuncAnimation

DIRECTORY = os.path.abspath(os.path.dirname(__file__))
PATH = f"{DIRECTORY}/../weights"
FILENAME = "tinynet.pt"

matplotlib.use("TkAgg")  # fow WSL


class TinyNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(28 * 28, 512)
        self.drop_1 = nn.Dropout(0.5)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(512, 512)
        self.relu_2 = nn.ReLU()
        self.drop_2 = nn.Dropout(0.5)
        self.linear_3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.drop_1(self.relu_1(self.linear_1(x)))
        x = self.drop_2(self.relu_2(self.linear_2(x)))
        return self.linear_3(x)


def preprocess(example):
    arr = np.array(example["input"])
    example["input"] = torch.from_numpy(arr).flatten() / 255.0
    return example


def collate(examples):
    inputs = torch.vstack([ex["input"] for ex in examples])
    labels = torch.vstack([ex["label"] for ex in examples])
    return inputs.squeeze().float(), labels.squeeze().long()


def flat_concat_params(model):
    return torch.concat(
        [
            param.detach().flatten()
            for param in model.parameters()
            if isinstance(param, nn.Parameter)
        ]
    )


def apply_perturbation(model, theta, delta, eta, alpha, beta):
    pert = theta + delta * alpha + beta * eta
    start = 0
    with torch.no_grad():
        for param in model.parameters():
            end = start + param.numel()
            tensor = pert[start:end]
            param.copy_(tensor.reshape(tuple(param.size())))
            start = end


def project_onto(d, w):
    return torch.dot(d, w) / d.norm()


def main():
    training = True if len(sys.argv) > 1 and sys.argv[1] == "--train".strip() else False

    size = 1000
    indices = np.random.choice(size, size, replace=False)
    mnist = load_dataset("mnist", trust_remote_code=True)

    trainset, testset = mnist.get("train").select(indices), mnist.get("test").select(
        indices
    )
    trainset, testset = trainset.rename_column("image", "input"), testset.rename_column(
        "image", "input"
    )
    trainset, testset = trainset.map(preprocess, num_proc=4), testset.map(
        preprocess, num_proc=4
    )
    trainset.set_format(type="torch", columns=["input", "label"])
    testset.set_format(type="torch", columns=["input", "label"])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = TinyNet()
    loss_fn = nn.CrossEntropyLoss()

    dataloader = data.DataLoader(
        trainset,
        batch_size=32,
        drop_last=False,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=4,
    )

    testloader = data.DataLoader(
        trainset,
        batch_size=size,
        drop_last=False,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=4,
    )

    if training:
        epochs = 150
        optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-3)
        _, _ = train(
            model,
            optimizer,
            loss_fn,
            dataloader,
            testloader,
            epochs=epochs,
            device=device,
            verbose=True,
        )

        if not os.path.exists(PATH):
            os.mkdir(PATH)
        torch.save(model.cpu().state_dict(), f=f"{PATH}/{FILENAME}")
        print("Model saved")

    else:
        state_dict = torch.load(f"{PATH}/{FILENAME}", map_location="cpu")
        model.load_state_dict(state_dict)
        print("Model loaded")

    theta = flat_concat_params(model).to(device)
    delta = torch.randn_like(theta).to(device)
    eta = torch.randn_like(theta).to(device)

    points = 100
    A = np.linspace(-1, 1, points)
    B = np.linspace(-1, 1, points)
    a, b = np.meshgrid(A, B)

    a_tensor, b_tensor = torch.from_numpy(a).float().to(device), torch.from_numpy(
        b
    ).float().to(device)
    z = torch.zeros((points, points)).to(device)

    for i in range(points):
        for j in range(points):
            alpha, beta = a_tensor[i, j], b_tensor[i, j]
            apply_perturbation(model, theta, delta, eta, alpha, beta)
            z[i, j] = test(model, loss_fn, testloader, device, verbose=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    param_range = (-1.5, 1.5)
    z = z.cpu().numpy()
    loss_range = (z.min() - 0.5, z.max() + 0.5)

    plt.xlabel("a")
    plt.ylabel("b")
    ax.set_zlabel("loss")
    ax.set_xlim(param_range)
    ax.set_ylim(param_range)
    ax.set_zlim(loss_range)
    ax.plot_surface(a, b, z, cmap="viridis", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
