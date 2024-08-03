import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, List

plt.style.use("dark_background")


def plot_metrics(
    metrics: Dict[str, Sequence[float]],
    colors: Optional[List[str]] = None,
    path: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Epoch",
    ylabel: str = "Value",
    figsize: tuple = (12, 8),
) -> None:
    plt.figure(figsize=figsize)

    if colors is None:
        colors = [
            "dodgerblue",
            "red",
            "aquamarine",
            "violet",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

    for (metric_name, values), color in zip(metrics.items(), colors):
        epochs = range(1, len(values) + 1)
        final_value = values[-1]
        plt.plot(epochs, values, label=f"{metric_name}: {final_value:.4f}", color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.style.use("dark_background")

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()


def load_metrics(file_path):
    with open(file_path, "rb") as file:
        train_loss, val_loss = pickle.load(file)
    return train_loss, val_loss


if __name__ == "__main__":
    # Generate some dummy data
    epochs = 100
    dummy_train_loss = np.random.rand(epochs) * 0.5 + 0.5
    dummy_val_loss = np.random.rand(epochs) * 0.3 + 0.7
    dummy_train_acc = np.linspace(0.5, 0.95, epochs) + np.random.randn(epochs) * 0.02
    dummy_val_acc = np.linspace(0.4, 0.9, epochs) + np.random.randn(epochs) * 0.02

    # Clip accuracies to be between 0 and 1
    dummy_train_acc = np.clip(dummy_train_acc, 0, 1)
    dummy_val_acc = np.clip(dummy_val_acc, 0, 1)

    # Create the metrics dictionary
    dummy_metrics = {
        "Training Loss": dummy_train_loss,
        "Validation Loss": dummy_val_loss,
        "Training Accuracy": dummy_train_acc,
        "Validation Accuracy": dummy_val_acc,
    }

    # Test the function
    plot_metrics(
        metrics=dummy_metrics,
        colors=["blue", "red", "green", "orange"],
        path="dummy_metrics_plot.png",
        title="Dummy Training and Validation Metrics",
        xlabel="Epoch",
        ylabel="Loss/Accuracy",
    )

    # Test with default colors and without saving
    plot_metrics(
        metrics=dummy_metrics, title="Dummy Metrics (Default Colors)", ylabel="Value"
    )

    # Test with subset of metrics
    subset_metrics = {
        "Training Loss": dummy_train_loss,
        "Validation Loss": dummy_val_loss,
    }

    plot_metrics(
        metrics=subset_metrics,
        colors=["purple", "orange"],
        title="Dummy Loss Metrics",
        ylabel="Loss",
    )
