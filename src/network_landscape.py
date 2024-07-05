import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

from scripts.utils import train
from datasets import load_dataset
from matplotlib.animation import FuncAnimation
from typing import Tuple

TRAIN = True
DIRECTORY = os.path.dirname(__file__)

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


def main():
    size = 1000
    mnist = load_dataset("mnist", trust_remote_code=True)
    trainset = mnist.get("train").select(np.random.choice(size, size, replace=False))
    trainset = trainset.rename_column("image", "input")
    trainset = trainset.map(preprocess, num_proc=4)
    trainset.set_format(type="torch", columns=["input", "label"])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = TinyNet()

    if TRAIN:

        epochs = 150
        optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-3)
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

        train_loss, _ = train(
            model,
            optimizer,
            loss_fn,
            dataloader,
            epochs=epochs,
            device=device,
            verbose=True,
        )


if __name__ == "__main__":
    main()
