import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from typing import Tuple, List, Optional, Callable


def create_dataloader(dataset: data.TensorDataset, batch_size: int) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    epochs: int = 10,
    augment_fn: Optional[Callable[..., torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    if verbose:
        print("Training has started")

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            if augment_fn is not None:
                inputs = augment_fn(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if verbose and epoch and ((epoch + 1) % int(epochs * 0.25) == 0):
            print(
                f"Epoch {epoch + 1} complete, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}"
            )

    if verbose:
        print("Training is complete")

    return train_losses, val_losses
