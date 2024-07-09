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
        modelpath: Optional[str] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.write = write
        self.filepath = filepath
        self.modelpath = modelpath

    def train(
        self,
        trainloader: data.DataLoader,
        testloader: Optional[data.DataLoader] = None,
        epochs: int = 10,
        device: Optional[str] = None,
        printevery: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        verbose = bool(printevery is not None and printevery)
        if verbose:
            print("Training started...")
        metrics = {
            "train loss": [],
            "test loss": [],
            "train acc": [],
            "test acc": [],
            "trajectory": [],
        }
        batches = len(trainloader)
        self.model.to(device)

        for epoch in range(epochs):
            self.model.train()
            eptrloss, eptracc = 0, 0

            if self.write:
                metrics["trajectory"].append(
                    [
                        p.cpu().flatten().detach().numpy()
                        for p in self.model.parameters()
                        if p.requires_grad
                    ]
                )

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.lossfn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                eptrloss += loss.item()
                eptracc += self.accuracy(outputs, labels, device=device)

            metrics["train loss"].append(eptrloss / batches)
            metrics["train acc"].append(eptracc / batches)

            if testloader is not None:
                eptsloss, eptsacc = self.eval(testloader, device=device, verbose=False)
                metrics["test loss"].append(eptsloss)
                metrics["test acc"].append(eptsacc)

            if printevery is not None and (epoch + 1) % printevery == 0 and epoch:
                s = f"Epoch {epoch + 1} complete, train loss: {metrics['train loss'][-1]:.4f}, train acc: {metrics['train acc'][-1]:.2f}"
                if testloader is not None:
                    s += f", test loss: {metrics['test loss'][-1]:.4f}, test acc: {metrics['test acc'][-1]:.2f}"
                print(s)

        if verbose:
            print("Training complete")

        if self.write:
            self.writetofiles(metrics, verbose=verbose)
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
                loss = self.lossfn(outputs, labels)
                testloss += loss.item()
                acc = self.accuracy(outputs, labels, device=device)
                testacc += acc

            testloss /= len(testloader)
            testacc /= len(testloader)

            if verbose:
                print(
                    f"Testing complete, test loss: {testloss:.4f}, test acc: {testacc:.2f}"
                )
            return testloss, testacc

    def accuracy(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        device: Optional[str] = None,
    ) -> float:
        assert outputs.size(0) == labels.size(0)

        with torch.no_grad():
            outputs, labels = outputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = torch.argmax(outputs, dim=-1)
            return (outputs == labels).float().mean().item()

    def writetofiles(
        self, metrics: Dict[str, List[float]], verbose: bool = False
    ) -> None:
        filepath = self.filepath if self.filepath is not None else "./metrics.h5"
        modelpath = (
            self.modelpath
            if self.modelpath is not None
            else f"./{self.model.__class__.__name__.lower()}.pt"
        )
        if verbose:
            print(f"Writing metrics to: {filepath}\nWriting weights to: {modelpath}")

        if not os.path.exists(os.path.dirname(filepath)):
            os.mkdir(os.path.dirname(filepath))
        if not os.path.exists(os.path.dirname(modelpath)):
            os.mkdir(os.path.dirname(modelpath))

        with h5py.File(f"{filepath}", mode="w") as file:
            metgroup = file.create_group("metrics")
            for key, metric in metrics.items():
                # TODO Fix
                metgroup.create_dataset(key, data=np.array(metric), dtype=np.float32)

        statedict = self.model.cpu().state_dict()
        torch.save(statedict, f"{modelpath}")

        if verbose:
            print("Metrics and weights written")
