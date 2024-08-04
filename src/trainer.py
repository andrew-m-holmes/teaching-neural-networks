import os
import h5py
import numpy as np
import torch
import torch.utils.data as data

from typing import Union, List, Callable, Optional, Dict
from .model import Model


class Trainer:

    def __init__(
        self,
        model: Model,
        optim: torch.optim.optimizer.Optimizer,
        loss_fn: Callable[..., torch.Tensor],
        dataloader: data.DataLoader,
        eval_dataloader: data.DataLoader,
        device: Optional[str] = None,
        path: Optional[str] = None,
        verbose: Optional[Union[bool, int]] = None,
    ) -> None:

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if verbose is None or verbose <= 0:
            verbose = False
        if path is not None:
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)

        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.path = path
        self.verbose = verbose

    def train(self, epochs: int = 1) -> Dict[str, List[float]]:

        self.model.to(self.device)

        metrics = {
            "train_losses": [],
            "test_losses": [],
            "train_accs": [],
            "test_accs": [],
        }

        if self.path is not None:
            self._write_trajectory(epoch=0)

        for epoch in range(epochs):

            self.model.train()
            epoch_train_loss, epoch_test_loss = 0, 0
            epoch_test_acc, epoch_test_acc = 0, 0

            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optim.zero_grad()
                logits = self.model(inputs)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optim.step()

            epoch_train_loss, epoch_train_acc = self.evaluate(self.dataloader).values()
            epoch_test_loss, epoch_test_acc = self.evaluate(
                self.eval_dataloader
            ).values()

            metrics["train_losses"].append(epoch_train_loss)
            metrics["test_losses"].append(epoch_test_loss)
            metrics["train_accs"].append(epoch_train_acc)
            metrics["test_accs"].append(epoch_test_acc)

            if self.path is not None:
                self._write_trajectory(epoch + 1)

        if self.path is not None:
            self._write_metrics(metrics)

        return metrics

    def evaluate(self, dataloader: data.DataLoader) -> Dict[str, float]:

        with torch.no_grad():
            self.model.eval()

            n_batches = len(dataloader)
            n_samples = 0
            net_correct = 0
            net_loss = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(**inputs).get("logits")
                loss = self.loss_fn(logits, labels)
                net_loss += loss.item()

                correct = torch.argmax(logits, dim=-1).eq(labels).sum()
                net_correct += correct.item()
                n_samples += labels.size(0)

            eval_acc = net_correct / n_samples
            eval_loss = net_loss / n_batches

            return {"eval_loss": eval_loss, "eval_acc": eval_acc}

    def _write_trajectory(self, epoch: int) -> None:
        if self.path is None:
            raise RuntimeError("'path' is None")

        weights = self.model.get_flat_weights()
        with h5py.File(self.path, mode="a") as file:
            if not epoch:
                file.create_group("trajectory")
            trajectory_group = file.get("trajectory")
            assert isinstance(trajectory_group, h5py.Group)
            trajectory_group.create_dataset(
                name=f"weights-epoch-{epoch}", data=weights, dtype=np.float32
            )

    def _write_metrics(self, metrics: Dict[str, List[float]]) -> None:
        if self.path is None:
            raise RuntimeError("'path' is None")

        with h5py.File(self.path, mode="a") as file:
            metrics_group = file.create_group("metrics")
            for name, metric in metrics.items():
                metrics_group.create_dataset(
                    name=name, data=np.array(metric), dtype=np.float32
                )
