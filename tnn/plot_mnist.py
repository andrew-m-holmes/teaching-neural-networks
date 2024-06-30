import pickle
import matplotlib.pyplot as plt
import os


def plot_loss(train_loss, val_loss, path):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]

    plt.plot(
        epochs,
        train_loss,
        label=f"Training Loss: {final_train_loss:.4f}",
        color="dodgerblue",
    )
    plt.plot(
        epochs,
        val_loss,
        label=f"Testing Loss: {final_val_loss:.4f}",
        color="red",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.style.use("dark_background")
    plt.savefig(path, dpi=300)


def plot_losses(losses, batch_sizes, path):
    colors = ["dodgerblue", "red", "blueviolet", "aquamarine", "coral"]
    epochs = range(1, len(next(iter(losses.values()))) + 1)

    final_losses = {batch_size: loss[-1] for batch_size, loss in losses.items()}
    best_batch_size = min(final_losses, key=final_losses.get)

    plt.figure(figsize=(12, 8))
    for i, batch_size in enumerate(batch_sizes):
        color = colors[i % len(colors)]
        linestyle = "-" if batch_size == best_batch_size else "--"
        final_loss = final_losses[batch_size]
        plt.plot(
            epochs,
            losses[batch_size],
            label=f"Batch {batch_size} Loss: {final_loss:.4f}",
            color=color,
            linestyle=linestyle,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")

    plt.style.use("dark_background")
    plt.savefig(path, dpi=300)


def load_metrics(file_path):
    with open(file_path, "rb") as file:
        train_loss, val_loss = pickle.load(file)
    return train_loss, val_loss


def main():
    path = os.path.abspath(os.path.dirname(__file__))
    print(path)
    plt.style.use("dark_background")

    batch_train_loss, batch_val_loss = load_metrics(f"{path}/metrics/batch_metrics.pkl")

    sgd_train_loss, sgd_val_loss = load_metrics(f"{path}/metrics/sgd_metrics.pkl")

    mini_batch_512_train_loss, mini_batch_512_val_loss = load_metrics(
        f"{path}/metrics/mini-batch_512_metrics.pkl"
    )

    plot_loss(
        batch_train_loss, batch_val_loss, f"{path}/../images/batch_loss_metrics.png"
    )
    plot_loss(sgd_train_loss, sgd_val_loss, f"{path}/../images/sgd_loss_metrics.png")
    plot_loss(
        mini_batch_512_train_loss,
        mini_batch_512_val_loss,
        f"{path}./images/mini-batch_512_loss_metrics.png",
    )

    mini_batch_256_train_loss, mini_batch_256_val_loss = load_metrics(
        f"{path}/metrics/mini-batch_256_metrics.pkl"
    )
    mini_batch_128_train_loss, mini_batch_128_val_loss = load_metrics(
        f"{path}/metrics/mini-batch_128_metrics.pkl"
    )
    mini_batch_64_train_loss, mini_batch_64_val_loss = load_metrics(
        f"{path}/metrics/mini-batch_64_metrics.pkl"
    )
    mini_batch_32_train_loss, mini_batch_32_val_loss = load_metrics(
        f"{path}/metrics/mini-batch_32_metrics.pkl"
    )

    train_losses = {
        512: mini_batch_512_train_loss,
        256: mini_batch_256_train_loss,
        128: mini_batch_128_train_loss,
        64: mini_batch_64_train_loss,
        32: mini_batch_32_train_loss,
    }

    val_losses = {
        512: mini_batch_512_val_loss,
        256: mini_batch_256_val_loss,
        128: mini_batch_128_val_loss,
        64: mini_batch_64_val_loss,
        32: mini_batch_32_val_loss,
    }

    batch_sizes = [512, 256, 128, 64, 32]

    plot_losses(
        train_losses, batch_sizes, f"{path}/../images/mini-batch_train_loss_comp.png"
    )
    plot_losses(
        val_losses, batch_sizes, f"{path}/../images/mini-batch_val_loss_comp.png"
    )


if __name__ == "__main__":
    main()
