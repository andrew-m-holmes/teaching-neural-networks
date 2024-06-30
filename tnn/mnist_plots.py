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
    plt.show()


def plot_training_times(batch_time, sgd_time, mini_batch_time, path):
    methods = ["Batch", "SGD", "Mini-batch"]
    times = [batch_time, sgd_time, mini_batch_time]
    colors = ["red", "dodgerblue", "blueviolet"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, times, color=colors)

    plt.xlabel("Gradient Descent Method")
    plt.ylabel("Time (seconds)")
    plt.yscale("log")
    plt.style.use("dark_background")

    plt.yticks(
        [1, 10, 100, 1000, 10000, 100000], ["1", "10", "100", "1k", "10k", "100k"]
    )

    for bar, time in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{time:.0f}",
            ha="center",
            va="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

    plt.savefig(path, dpi=300)
    plt.show()


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
    plt.show()


def load_metrics(file_path):
    with open(file_path, "rb") as file:
        train_loss, val_loss = pickle.load(file)
    return train_loss, val_loss


def load_times(file_path):
    with open(file_path, "rb") as file:
        elasped_time = pickle.load(file)
    return elasped_time


def main():

    path = os.path.abspath(os.path.dirname(__file__))

    plt.style.use("dark_background")

    batch_train_loss, batch_val_loss = load_metrics(f"{path}/data/batch_metrics.pkl")
    batch_time = load_times(f"{path}/data/batch_time.pkl")

    sgd_train_loss, sgd_val_loss = load_metrics(f"{path}/data/stochastic_metrics.pkl")
    sgd_time = load_times(f"{path}/data/stochastic_time.pkl")

    mini_batch_512_train_loss, mini_batch_512_val_loss = load_metrics(
        f"{path}/data/mini_batch_512_metrics.pkl"
    )
    mini_batch_512_time = load_times(f"{path}/data/mini_batch_512_time.pkl")

    plot_loss(batch_train_loss, batch_val_loss, f"{path}/../images/batch_loss_metrics.png")
    plot_loss(sgd_train_loss, sgd_val_loss, f"{path}/../images/sgd_loss_metrics.png")
    plot_loss(
        mini_batch_512_train_loss,
        mini_batch_512_val_loss,
        f"{path}./images/mini_batch_512_loss_metrics.png",
    )

    plot_training_times(
        batch_time, sgd_time, mini_batch_512_time, f"{path}/../images/training_times.png"
    )

    mini_batch_256_train_loss, mini_batch_256_val_loss = load_metrics(
        f"{path}/data/mini_batch_256_metrics.pkl"
    )
    mini_batch_128_train_loss, mini_batch_128_val_loss = load_metrics(
        f"{path}/data/mini_batch_128_metrics.pkl"
    )
    mini_batch_64_train_loss, mini_batch_64_val_loss = load_metrics(
        f"{path}/data/mini_batch_64_metrics.pkl"
    )
    mini_batch_32_train_loss, mini_batch_32_val_loss = load_metrics(
        f"{path}/data/mini_batch_32_metrics.pkl"
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

    plot_losses(train_losses, batch_sizes, f"{path}/../images/mini_batch_train_loss_comp.png")
    plot_losses(val_losses, batch_sizes, f"{path}/../images/mini_batch_val_loss_comp.png")


if __name__ == "__main__":
    main()
