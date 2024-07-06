import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from copy import deepcopy
from src.utils import test
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


def genrandvecs(model):
    thetas = [p.detach().clone().flatten() for p in model.parameters()]
    deltas, etas = [], []

    for theta in thetas:
        vecx, vecy = filternorm(theta), filternorm(theta)
        deltas.append(vecx)
        etas.append(vecy)

    assert len(thetas) == len(deltas) == len(etas)
    return thetas, deltas, etas


def filternorm(flatparams):
    vec = torch.randn_like(flatparams)
    vec *= flatparams.norm() / vec.norm()
    return vec


def genminimizer(thetas, deltas, etas, alpha, beta):
    thetahats = []

    for theta, delta, eta in zip(thetas, deltas, etas):
        assert theta.numel() == delta.numel() == eta.numel()
        newflatparams = theta.detach().flatten() + alpha * delta + beta * eta
        thetahats.append(newflatparams)

    return thetahats


def evaluate(model, origin, thetahats, lossfn, testloader, device=None):
    setparams(model, thetahats)
    loss = test(model, lossfn, testloader, device=device)
    setparams(model, origin)
    return loss


def setparams(model, thetas):
    assert len(list(model.parameters())) == len(thetas)
    for param, theta in zip(model.parameters(), thetas):
        assert param.numel() == theta.numel()
        param.data = theta.reshape(param.size())


def main():

    samples = 100
    testset = load_dataset("ylecun/mnist", split="test").select(
        np.random.choice(samples, samples, replace=False)
    )
    testset = testset.rename_column("image", "input")
    testset = testset.map(preprocess, num_proc=4)
    testset.set_format("torch", columns=["input", "label"])
    testloader = data.DataLoader(
        testset, batch_size=samples, pin_memory=True, num_workers=4, collate_fn=collate
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = TinyNet()
    model.load_state_dict(torch.load(f"{PATH}/{FILENAME}", map_location="cpu"))
    lossfn = nn.CrossEntropyLoss()
    thetas, deltas, etas = genrandvecs(model)

    resolution = 10
    a = torch.linspace(-1, 1, resolution)
    b = torch.linspace(-1, 1, resolution)
    A, B = torch.meshgrid(a, b, indexing="ij")
    Z = torch.ones((resolution, resolution))

    print("Landscape generating...")
    start = time.perf_counter()

    for i in range(resolution):
        for j in range(resolution):
            thetahats = genminimizer(thetas, deltas, etas, A[i][j], B[i][j])
            loss = evaluate(model, thetas, thetahats, lossfn, testloader, device=device)
            Z[i, j] = loss
            print(f"Iteration: {i * 10 + j}, loss: {loss:.4f}")

    end = time.perf_counter()
    print(f"Elasped time: {(end - start):.2f} seconds")

    a = A.numpy()
    b = B.numpy()
    z = Z.numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    ax.set_zlabel("loss")
    ax.plot_surface(a, b, z, cmap="viridis", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
