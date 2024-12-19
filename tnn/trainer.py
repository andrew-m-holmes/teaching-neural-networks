import os
import sys
import h5py
import tnn
import torch
import torch.nn as nn
import torch.utils.data as data
import time
import numpy as np
import logging

from datetime import timedelta, datetime
from typing import Union, List, Callable, Optional, Dict, Tuple, Any


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: Callable[..., torch.Tensor],
        dataloader: data.DataLoader,
        eval_dataloader: data.DataLoader,
        epochs: int = 100,
        store_update_metrics: bool = False,
        unpack_inputs: bool = False,
        device: Optional[str] = None,
        pin_memory: bool = False,
        non_blocking: bool = False,
        to_fn: Optional[Callable[..., Tuple[Any, Any]]] = None,
        path: Optional[str] = None,
        verbose: Optional[Union[bool, int]] = None,
        profile: bool = False,
        logger_name: Optional[str] = None,
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
        self.epochs = epochs
        self.store_update_metrics = store_update_metrics
        self.unpack_inputs = unpack_inputs
        self.device = device
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        self.to_fn = to_fn
        self.path = path
        self.verbose = verbose
        self.profile = profile

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        date = datetime.today().strftime("%Y-%m-%d:%H:%M:%S")
        log_file_prefix = f"{logger_name}-" if logger_name is not None else ""
        file_path = os.path.join(path, f"{date}-{log_file_prefix}trainer-logs.txt")

        file_handler = logging.FileHandler(file_path)
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler.setLevel(level=logging.INFO)
        stdout_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        stdout_handler.setFormatter(formatter)

        self.logger = logging.getLogger(name=logger_name)
        self.logger.setLevel(level=logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)

    def train(self) -> Dict[str, List[float]]:
        if self.path is not None:
            dirname = os.path.dirname(self.path)
            os.makedirs(dirname, exist_ok=True)

        self.model.to(self.device, non_blocking=self.non_blocking)
        if self.verbose:
            self.logger.info(f"model using {self.device}")
        if self.device != "cuda" and self.profile:
            self.logger.warning(f"cannot profile, profile only enabled for cuda")

        n_batches = len(self.dataloader)
        n_samples = sum([labels.size(0) for _, labels in self.dataloader])
        metrics = {
            "train_losses": [],
            "test_losses": [],
            "train_accs": [],
            "test_accs": [],
            "epoch_times": [],
        }
        if self.store_update_metrics:
            metrics["update_train_losses"] = []
            metrics["update_train_accs"] = []
            metrics["update_times"] = []

        if self.verbose:
            self.logger.info("training started")

        allocated, reserved, train_start_time = None, None, time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            epoch_train_loss, epoch_train_acc = 0, 0

            if self.profile and self.device == "cuda":
                allocated, reserved = 0, 0

            self.model.train()
            for inputs, labels in self.dataloader:
                update_start_time = time.time()
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
                correct = self._compute_correct(logits, labels)
                epoch_train_acc += correct
                update_end_time = time.time()

                if self.store_update_metrics:
                    update_acc = correct / labels.size(0)
                    update_duration = update_start_time - update_end_time
                    metrics["update_train_losses"].append(loss.item())
                    metrics["update_train_accs"].append(update_acc)
                    metrics["update_times"].append(update_duration)

                if allocated is not None and reserved is not None:
                    allocated += torch.cuda.memory_allocated(device=self.device)
                    reserved += torch.cuda.memory_reserved(device=self.device)

            epoch_train_loss /= n_batches
            epoch_train_acc /= n_samples
            epoch_test_loss, epoch_test_acc = self.evaluate(
                self.eval_dataloader
            ).values()

            if allocated is not None and reserved is not None:
                allocated /= n_batches
                reserved /= n_batches

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            total_duration = epoch_end_time - train_start_time

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
                    allocated,
                    reserved,
                )

        if self.verbose:
            self.logger.info("training complete")

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

        self.logger.info(
            f"(epoch: {epoch}/{self.epochs}): (train loss: {metrics['train_losses'][-1]:.4f}, test loss: {metrics['test_losses'][-1]:.4f}, train acc: {(metrics['train_accs'][-1] * 100):.2f}%, test acc: {(metrics['test_accs'][-1] * 100):.2f}%){profile_str}{time_str}"
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
                    self.logger.info(f"{name} saved to {self.path}/metrics/{name}")
