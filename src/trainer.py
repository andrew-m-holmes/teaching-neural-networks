import os
import h5py
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from typing import Tuple, List, Optional, Callable, Dict, Any


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lossfn: nn.modules.loss._Loss,
        write: bool = True,
        filepath: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.write = write
        self.filepath = filepath
        self.filename = filename

    def train(
        self,
        dataloader: data.DataLoader,
        testloader: Optional[data.DataLoader] = None,
        accfn: Optional[Callable[..., float]] = None,
        epochs: int = 10,
        device: Optional[str] = None,
        printevery: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        verbose = printevery is not None and printevery
        metrics = {"train loss": [], "test loss": [], "train acc": [], "testacc": []}
        batches = len(dataloader)
        if verbose:
            print("Training started...")

        for epoch in range(epochs):
            self.model.train()
            eptrloss, eptracc = 0, 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.lossfn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                eptrloss += loss.item()
                eptracc += self.precision(outputs, labels, islogits=True, device=device)

            metrics["train loss"].append(eptrloss / batches)
            metrics["train acc"].append(eptracc / batches)
            if testloader is not None:
                eptsloss, eptsacc = self.eval(testloader, device=device, verbose=False)
                metrics["test loss"].append(eptsloss)
                metrics["test acc"].append(eptsacc)

            if printevery is not None and (epoch + 1) % printevery and epoch:
                s = f"Epoch {epoch + 1} complete, train loss: {metrics['train loss'[-1]]:.4f}, train acc: {metrics['train acc'][-1]:.2f}"
                if testloader is not None:
                    s += f", test loss: {metrics['test loss'][-1]:.4f}, test acc: {metrics['test acc'][-1]:.2f}"
                print(s)

        if verbose:
            print("Training complete")

        if self.write:
            if verbose:
                print("Writing...")
            self.save(metrics)
            if verbose:
                print("Metrics saved")
        return metrics

    def eval(
        self,
        testloader: data.DataLoader,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        if verbose:
            print("Testing started...")
        self.model.eval()
        with torch.no_grad():
            testloss, testacc = 0, 0

            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                outputs = self.model(inputs)
                loss = self.lossfn(inputs, outputs)
                testloss += loss.item()
                acc = self.precision(outputs, labels, islogits=True, device=device)
                testacc += acc

            testloss /= len(testloader)
            testacc /= len(testloader)

            if verbose:
                print(
                    f"Testing complete, test loss: {testloss:.4f}, test acc: {testacc:.2f}"
                )
            return testloss, testacc

    def precision(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        islogits: bool = True,
        device: Optional[str] = None,
    ) -> float:
        assert outputs.size(0) == labels.size(0)

        with torch.no_grad():
            outputs, labels = outputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            if islogits:
                outputs = torch.argmax(outputs, dim=-1)
            return (outputs == labels).float().mean().item()

    def save(self, metrics: Dict[str, List[float]]) -> None:
        filepath = self.filepath if self.filepath is not None else "."
        filename = self.filename if self.filename is not None else "trainer-metrics"

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        with h5py.File(f"{filepath}/{filename}.h5", mode="w") as file:
            metgroup = file.create_group("metrics")
            for key, metric in metrics.values():
                metgroup.create_dataset(key, data=np.array(metric), dtype=np.float32)

    def dstodl(self, dataset: datasets.Dataset, **dlkwargs) -> data.DataLoader:
        dataloader = data.DataLoader(dataset, **dlkwargs)
        return dataloader
