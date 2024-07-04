import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from typing import Tuple, List, Optional


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: data.DataLoader,
    eval_dataloader: data.DataLoader,
    epochs: int = 10,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    if verbose:
        print("Training has started")

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        eval_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in eval_dataloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()

        train_loss /= len(train_dataloader)
        eval_loss /= len(eval_dataloader)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        if verbose and epoch and ((epoch + 1) % int(epochs * 0.25) == 0):
            print(
                f"Epoch {epoch + 1} complete, train loss: {train_loss:.3f}, eval loss: {eval_loss:.3f}"
            )
    if verbose:
        print("Training is complete")
    return train_losses, eval_losses
