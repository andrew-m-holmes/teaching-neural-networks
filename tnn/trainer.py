import os
import h5py
import tnn
import torch
import torch.utils.data as data
import time
import numpy as np

from torch.optim.lr_scheduler import LRScheduler
from datetime import timedelta
from .model import Model
from typing import Union, List, Callable, Optional, Dict, Tuple, Any


class Trainer:

    def __init__(
        self,
        model: Model,
        optim: torch.optim.Optimizer,
        loss_fn: Callable[..., torch.Tensor],
        dataloader: data.DataLoader,
        eval_dataloader: data.DataLoader,
        scheduler: Optional[LRScheduler] = None,
        epochs: int = 100,
        unpack_inputs: bool = False,
        device: Optional[str] = None,
        pin_memory: bool = False,
        non_blocking: bool = False,
        to_fn: Optional[Callable[..., Tuple[Any, Any]]] = None,
        path: Optional[str] = None,
        verbose: Optional[Union[bool, int]] = None,
        profile: bool = False,
    ) -> None:

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        if device != "cuda":
            pin_memory = False
            non_blocking = False

        if to_fn is None:
            to_fn = lambda inputs, labels, device, non_blocking: (
                inputs.to(device=device, non_blocking=non_blocking),
                labels.to(device=device, non_blocking=non_blocking),
            )

        if not verbose or verbose < 0:
            verbose = False
        else:
            verbose = int(verbose)

        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.epochs = epochs
        self.unpack_inputs = unpack_inputs
        self.device = device
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        self.to_fn = to_fn
        self.path = path
        self.verbose = verbose
        self.profile = profile

    def train(self) -> Dict[str, List[float]]:
        if self.path is not None:
            dirname = os.path.dirname(self.path)
            os.makedirs(dirname, exist_ok=True)

        self.model.to(self.device, non_blocking=self.non_blocking)
        if self.verbose:
            print(f"model using {self.device}")
        if self.device != "cuda" and self.profile:
            print(f"cannot profile, profile only enabled for cuda")

        n_batches = len(self.dataloader)
        metrics = {
            "train_losses": [],
            "test_losses": [],
            "train_accs": [],
            "test_accs": [],
            "epoch_times": [],
        }
        epoch_allocated, epoch_reserved, start_time = None, None, None

        if self.verbose:
            print("training started")

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            if start_time is None:
                start_time = epoch_start_time
            epoch_train_loss, epoch_train_acc, n_samples = 0, 0, 0

            if self.profile and self.device == "cuda":
                epoch_allocated, epoch_reserved = 0, 0

            self.model.train()
            for inputs, labels in self.dataloader:
                inputs, labels = self.to_fn(
                    inputs, labels, device=self.device, non_blocking=self.non_blocking
                )

                self.optim.zero_grad()
                logits = (
                    self.model(**inputs).get("logits")
                    if self.unpack_inputs
                    else self.model(inputs).get("logits")
                )
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optim.step()

                epoch_train_loss += loss.item()
                epoch_train_acc += self._compute_correct(logits, labels)
                n_samples += labels.size(0)

                if epoch_allocated is not None and epoch_reserved is not None:
                    epoch_allocated += torch.cuda.memory_allocated(device=self.device)
                    epoch_reserved += torch.cuda.memory_reserved(device=self.device)

            epoch_train_loss /= n_batches
            epoch_train_acc /= n_samples
            epoch_test_loss, epoch_test_acc = self.evaluate(
                self.eval_dataloader
            ).values()
            if epoch_allocated is not None and epoch_reserved is not None:
                epoch_allocated /= n_batches
                epoch_reserved /= n_batches

            if self.scheduler is not None:
                self.scheduler.step(epoch_test_loss)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            total_duration = epoch_end_time - start_time
            metrics["train_losses"].append(epoch_train_loss)
            metrics["test_losses"].append(epoch_test_loss)
            metrics["train_accs"].append(epoch_train_acc)
            metrics["test_accs"].append(epoch_test_acc)
            metrics["epoch_times"].append(epoch_duration)

            print_info = bool(
                self.verbose
                and ((epoch + 1) % self.verbose == 0 or (epoch + 1) == self.verbose)
            )

            if print_info:
                self._epoch_print(
                    epoch + 1,
                    epoch_duration,
                    total_duration,
                    metrics,
                    epoch_allocated,
                    epoch_reserved,
                )

        if self.verbose:
            print("training complete")

        if self.path is not None:
            self._write_metrics(metrics, verbose=bool(self.verbose))

        return metrics

    def evaluate(self, dataloader: data.DataLoader) -> Dict[str, float]:
        with torch.no_grad():
            self.model.eval()

            n_batches = len(dataloader)
            n_samples = sum([labels.size(0) for _, labels in dataloader])
            net_loss = 0
            net_correct = 0

            for inputs, labels in dataloader:
                inputs, labels = self.to_fn(
                    inputs, labels, self.device, self.non_blocking
                )

                logits = (
                    self.model(**inputs).get("logits")
                    if self.unpack_inputs
                    else self.model(inputs).get("logits")
                )

                loss = self.loss_fn(logits, labels)
                correct = self._compute_correct(logits, labels)
                net_loss += loss.item()
                net_correct += correct

            eval_loss = net_loss / n_batches
            eval_acc = net_correct / n_samples
            return {"eval_loss": eval_loss, "eval_acc": eval_acc}

    def _compute_correct(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        correct = torch.argmax(logits, dim=-1).eq(labels).sum()
        return correct.item()

    def _epoch_print(
        self,
        epoch: int,
        epoch_duration: float,
        total_duration: float,
        metrics: Dict[str, List[float]],
        allocated: Optional[float] = None,
        reserved: Optional[float] = None,
    ) -> None:
        profile_str = (
            f"\n(gpu memory profile): (average allocated: {allocated // 1.024e6} MB, average reserved: {reserved // 1.024e6} MB)"
            if allocated is not None and reserved is not None
            else ""
        )

        epoch_time_str = str(timedelta(seconds=int(epoch_duration)))
        elapsed_time_str = str(timedelta(seconds=int(total_duration)))

        time_str = f"\n(duration info): (epoch duration: {epoch_time_str}, elapsed time: {elapsed_time_str})"

        lr_str = (
            f"\n(learning rate: {self.optim.param_groups[0]["lr"]:.1e})"
            if self.scheduler is not None
            else ""
        )

        print(
            f"(epoch: {epoch}/{self.epochs}): (train loss: {metrics['train_losses'][-1]:.4f}, test loss: {metrics['test_losses'][-1]:.4f}, train acc: {(metrics['train_accs'][-1] * 100):.2f}%, test acc: {(metrics['test_accs'][-1] * 100):.2f}%){lr_str}{profile_str}{time_str}"
        )

    def _write_metrics(
        self, metrics: Dict[str, List[float]], verbose: bool = False
    ) -> None:
        if self.path is None:
            raise RuntimeError("'path' is None")

        with h5py.File(self.path, mode="a") as file:
            metrics_group = tnn._get_group("metrics", file, clear=True)
            for name, metric in metrics.items():
                metrics_group.create_dataset(
                    name=name, data=np.array(metric), dtype=np.float32
                )

                if verbose:
                    print(f"{name} saved to {self.path}/metrics/{name}")
