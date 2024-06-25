import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class polymod:

    def __init__(self, ndim: int) -> None:
        self.coefficients = np.random.randint(-1, 1, ndim).astype(
            float
        ) * np.random.randn(ndim)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.sum(input * self.coefficients, axis=1)


def main():

    np.random.seed(0)
    ndim = 5
    data = np.random.randn(100, ndim)

    pca = PCA(n_components=2, svd_solver="covariance_eigh")
    model = polymod(ndim)
    pca.fit(model.coefficients.reshape(1, -1))
=======
import torch
import torch.nn as nn
from datasets import load_dataset


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(784, 512)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(512, 512)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu_1(self.linear_1(x))
        x = self.relu_2(self.linear_2(x))
        return self.linear_3(x)


def normalize(example):
    arr = np.array(example["image"]) / 255
    example["input"] = arr
    return example


def main():
    trainset = load_dataset("ylecun/mnist", split="train")
    size = 1000
    indices = np.random.choice(len(trainset), size, replace=False)
    sampleset = trainset.select(indices)
    sampleset = sampleset.map(
        normalize, batched=True, batch_size=64, remove_columns="image"
    )
    inputs = torch.tensor(sampleset["input"]).reshape(-1, 28 * 28)
    labels = torch.tensor(sampleset["label"]).long()

    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    inputs, loss = inputs.numpy(), loss.numpy()


if __name__ == "__main__":
    main()
>>>>>>> a621170315fcd94b69b23735eb31f9eb3fa853f4
