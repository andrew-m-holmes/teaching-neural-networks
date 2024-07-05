import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from typing import Tuple, List, Optional


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.modules.loss._Loss,
    trainloader: data.DataLoader,
    testloader: Optional[data.DataLoader] = None,
    epochs: int = 10,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    if verbose:
        print("Training has started")

    model = model.to(device)
    train_losses = []
    eval_losses = []
    eval_loss = None

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if testloader is not None:
            eval_loss = test(model, loss_fn, testloader, device, verbose=False)
            eval_losses.append(eval_loss)

        train_loss /= len(trainloader)
        train_losses.append(train_loss)
        if verbose and epoch and ((epoch + 1) % int(epochs * 0.25) == 0):
            print(
                f"Epoch {epoch + 1} complete, train loss: {train_loss:.3f}"
                + (f", eval loss: {eval_loss:.3f}" if eval_loss is not None else "")
            )
    if verbose:
        print("Training is complete")
    return train_losses, eval_losses


def test(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    testloader: data.DataLoader,
    device: Optional[str] = None,
    verbose: bool = False,
) -> float:
    if verbose:
        print("Testing started")
    test_loss = 0
    model.eval().to(device)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

    test_loss /= len(testloader)
    print(f"Testing complete, loss: {test_loss:.4f}")
    return test_loss
