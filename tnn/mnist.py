import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import torch.optim as optim
import argparse
from datasets import load_dataset
import tnn.utils as utils


def preprocess(example):
    arr = np.reshape(example["input"], -1)
    example["input"] = arr
    return example


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(28 * 28, 512)
        self.norm_1 = nn.LayerNorm(512)
        self.drop_1 = nn.Dropout(p=0.4)
        self.linear_2 = nn.Linear(512, 512)
        self.norm_2 = nn.LayerNorm(512)
        self.drop_2 = nn.Dropout(p=0.2)
        self.linear_3 = nn.Linear(512, 512)
        self.norm_3 = nn.LayerNorm(512)
        self.linear_4 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.drop_1(f.relu(self.norm_1(self.linear_1(x))))
        x = self.drop_2(f.relu(self.norm_2(self.linear_2(x))))
        x = f.relu(self.norm_3(self.linear_3(x)))
        out = self.linear_4(x)
        return out


def main():
    parser = argparse.ArgumentParser(description="SGD Variations")
    parser.add_argument(
        "--variant", type=str, choices=["batch", "mini-batch", "sgd"], required=True
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    mnist = load_dataset("mnist", trust_remote_code=True)
    train, test = mnist.get("train"), mnist.get("test")

    train.set_format(type="numpy", columns=["image", "label"])
    test.set_format(type="numpy", columns=["image", "label"])

    num_train_samples = 10000
    num_test_samples = 1000
    train_indices = np.random.choice(
        num_train_samples, num_train_samples, replace=False
    )
    test_indices = np.random.choice(num_test_samples, num_test_samples, replace=False)
    train = train.rename_column("image", "input").select(train_indices)
    test = test.rename_column("image", "input").select(test_indices)

    train = train.map(preprocess)
    test = test.map(preprocess)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_inputs = torch.from_numpy(train["input"]).float().squeeze()
    test_inputs = torch.from_numpy(test["input"]).float().squeeze()
    train_labels = torch.from_numpy(train["label"]).long()
    test_labels = torch.from_numpy(test["label"]).long()

    train_dataset = data.TensorDataset(train_inputs, train_labels)
    test_dataset = data.TensorDataset(test_inputs, test_labels)

    batch_size = None
    if args.variant == "batch":
        batch_size = len(train_dataset)
    elif args.variant == "mini-batch":
        batch_size = args.batch_size
    elif args.variant == "sgd":
        batch_size = 1

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0)
    if args.verbose:
        print(f"SGD Variant: {args.variant.capitalize()}")

    start_time = time.time()
    train_loss, test_loss = utils.train(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        epochs=args.epochs,
        device=device,
        verbose=args.verbose,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    if args.write:
        if args.verbose:
            print(f"Writing to {args.path}")
        path = args.path
        batch_size_tag = f"_{batch_size}" if args.variant == "mini-batch" else ""
        with open(f"{path}/{args.variant}{batch_size_tag}_metrics.pkl", "wb") as file:
            pickle.dump((train_loss, test_loss), file)
        with open(f"{path}/{args.variant}{batch_size_tag}_time.pkl", "wb") as file:
            pickle.dump(elapsed_time, file)


if __name__ == "__main__":
    main()
