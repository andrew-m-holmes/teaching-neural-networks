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


def paramflat(model):
    return torch.concat([p.clone().detach().flatten() for p in model.parameters()])


def filternorm(vector, parameters):
    vnorm = vector.norm()
    pnorm = parameters.norm()
    return vector * (pnorm / vnorm)


@torch.no_grad
def setparams(model, parameters, clone=True):
    if clone:
        model = deepcopy(model)
    start = 0
    for p in model.parameters():
        end = start + p.numel()
        p.copy_(parameters[start:end].reshape(*p.size()))
    return model


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
    theta = paramflat(model).to(device)
    directions = torch.randn_like(theta, device=device), torch.randn_like(
        theta, device=device
    )
    delta = filternorm(directions[0], theta)
    eta = filternorm(directions[1], theta)

    granulairty = 10
    A = torch.linspace(-0.1, 0.1, granulairty, device=device)
    B = torch.linspace(-0.1, 0.1, granulairty, device=device)
    alpha, beta = torch.meshgrid(A, B, indexing="ij")
    losses = torch.zeros(granulairty * granulairty)
    newparams = (
        theta.unsqueeze(0)
        + alpha.reshape(-1, 1) * delta.unsqueeze(0)
        + beta.reshape(-1, 1) * eta.unsqueeze(0)
    )

    print("Landscape generating...")
    start = time.perf_counter()
    torch.cuda.empty_cache()
    for i in range(newparams.size(0)):
        parameters = newparams[i]
        model = setparams(model, parameters, clone=False)
        loss = test(model, lossfn, testloader, device=device, verbose=False)
        losses[i] = loss
    # for i in range(granulairty):
    #     for j in range(granulairty):
    #         newparams = theta + alpha[i, j] * delta + beta[i, j] * eta
    #         model = setparams(model, newparams, clone=True)
    #         loss = test(model, lossfn, testloader, device=device, verbose=False)
    #         losses[i, j] = loss
    end = time.perf_counter()
    print(f"Elasped time: {(end - start):.2f} seconds")

    a = alpha.cpu().numpy()
    b = beta.cpu().numpy()
    z = losses.cpu().numpy().reshape(granulairty, granulairty)

    paramrange = (-0.2, 0.2)
    lossrange = (0.0, 2.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    ax.set_zlabel("loss")
    ax.set_xlim(paramrange)
    ax.set_ylim(paramrange)
    ax.set_zlim(lossrange)
    ax.plot_surface(a, b, z, cmap="viridis", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
