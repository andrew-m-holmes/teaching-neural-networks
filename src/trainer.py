import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from typing import Tuple, List, Optional, Callable, Dict


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable[..., torch.Tensor],
        write: bool = True,
        metric_path: Optional[str] = None,
        param_path: Optional[str] = None,
        traj_path: Optional[str] = None,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.write = write
        self.metric_path = metric_path
        self.param_path = param_path
        self.traj_path = traj_path

    def train(
        self,
        trainloader: data.DataLoader,
        testloader: Optional[data.DataLoader] = None,
        epochs: int = 10,
        device: Optional[str] = None,
        print_every: Optional[int] = None,
    ) -> Tuple[Dict[str, List[float]], torch.Tensor]:

        verbose = bool(print_every is not None and print_every)
        if verbose:
            print("Training started...")

        metrics = {
            "train loss": [],
            "test loss": [],
            "train acc": [],
            "test acc": [],
        }
        trajectory = []
        batches = len(trainloader)

        self.model.to(device)
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss, epoch_train_acc = 0, 0

            if self.write:
                trajectory.append(
                    torch.cat(
                        [
                            p.cpu().flatten().detach()
                            for p in self.model.parameters()
                            if p.requires_grad
                        ]
                    )
                )

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_acc += self.accuracy(outputs, labels, device=device)

            metrics["train loss"].append(epoch_train_loss / batches)
            metrics["train acc"].append(epoch_train_acc / batches)

            if testloader is not None:
                epoch_test_loss, epoch_test_acc = self.eval(
                    testloader, device=device, verbose=False
                )
                metrics["test loss"].append(epoch_test_loss)
                metrics["test acc"].append(epoch_test_acc)

            if (print_every and (epoch + 1) % print_every == 0 and epoch) or (
                epoch + 1
            ) == print_every:
                s = f"Epoch {epoch + 1} complete, train loss: {metrics['train loss'][-1]:.4f}, train acc: {metrics['train acc'][-1]:.2f}"
                if testloader is not None:
                    s += f", test loss: {metrics['test loss'][-1]:.4f}, test acc: {metrics['test acc'][-1]:.2f}"
                print(s)

        if verbose:
            print("Training complete")

        trajectory = torch.stack(trajectory)
        if self.write:
            self.write_files(metrics, trajectory, verbose=verbose)
        return metrics, trajectory

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
            test_loss, test_acc = 0, 0

            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                acc = self.accuracy(outputs, labels, device=device)
                test_acc += acc

            test_loss /= len(testloader)
            test_acc /= len(testloader)

            if verbose:
                print(
                    f"Testing complete, test loss: {test_loss:.4f}, test acc: {test_acc:.2f}"
                )
            return test_loss, test_acc

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

    def write_files(
        self,
        metrics: Dict[str, List[float]],
        trajectory: torch.Tensor,
        verbose: bool = False,
    ) -> None:
        metric_path = (
            self.metric_path if self.metric_path is not None else "./metrics.h5"
        )
        param_path = (
            self.param_path
            if self.param_path is not None
            else f"./{self.model.__class__.__name__.lower()}.pt"
        )
        traj_path = (
            self.traj_path
            if self.traj_path is not None
            else f"./{self.model.__class__.__name__}_traj.pt"
        )

        if verbose:
            print(
                f"Writing metrics to: {metric_path}\nWriting parameters to: {param_path}\nWriting trajectory to: {traj_path}"
            )

        if not os.path.exists(os.path.dirname(metric_path)):
            os.mkdir(os.path.dirname(metric_path))
        if not os.path.exists(os.path.dirname(param_path)):
            os.mkdir(os.path.dirname(param_path))

        with h5py.File(metric_path, mode="w") as file:
            metric_group = file.create_group("metrics")
            for key, metric in metrics.items():
                metric_group.create_dataset(
                    key, data=np.array(metric), dtype=np.float32
                )

        state_dict = self.model.cpu().state_dict()
        torch.save(state_dict, param_path)
        torch.save(trajectory, traj_path)

        if verbose:
            print("Metrics and weights written")
