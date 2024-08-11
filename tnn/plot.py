import pickle
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


def plot_surface_3D(
    mesh, cmap: str = "viridis", show: bool = True, file_path: Optional[str] = None
) -> None:

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.grid(False)

    ax.plot_surface(mesh, cmap=cmap, alpha=0.5)

    ax.set_xlabel("Principle Component 1", color="gray")
    ax.set_ylabel("Principle Component 2", color="gray")
    ax.set_zlabel("Loss", color="gray")

    ax.tick_params(axis="x", colors="gray")
    ax.tick_params(axis="y", colors="gray")
    ax.tick_params(axis="z", colors="gray")

    if file_path is not None:
        plt.savefig(file_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_contour(
    self,
    levels: int = 100,
    plot_trajectory: bool = False,
    cmap: str = "viridis",
    show: bool = True,
    file_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    contour = ax.contourf(*self.mesh, cmap=cmap, levels=levels, antialiased=True)
    if plot_trajectory:
        assert self.trajectory is not None
        x, y = self.trajectory
        ax.plot(x, y, marker=".", color="dodgerblue", linewidth=2, markersize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Principal Component 1", color="white")
    ax.set_ylabel("Principal Component 2", color="white")
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Loss", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path, facecolor="black", edgecolor="none")
    if show:
        plt.show()
    else:
        plt.close()


def animate_contour(
    self,
    levels: int = 100,
    fps: int = 5,
    cmap: str = "viridis",
    show: bool = True,
    file_path: Optional[str] = None,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    contour = ax.contourf(*self.mesh, cmap=cmap, levels=levels, antialiased=True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Principal Component 1", color="white")
    ax.set_ylabel("Principal Component 2", color="white")

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Loss", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    assert self.trajectory is not None
    trajectory = self.trajectory.T
    pc_0 = trajectory[0]
    pcx, pcy = [pc_0[0]], [pc_0[1]]
    (pathline,) = ax.plot(pcx, pcy, color="dodgerblue", lw=2)
    (point,) = ax.plot(pcx, pcy, "o", color="dodgerblue", markersize=8)
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

    def update(frame):
        pc = trajectory[frame]
        pcx.append(pc[0])
        pcy.append(pc[1])
        pathline.set_data(pcx, pcy)
        point.set_data([pcx[-1]], [pcy[-1]])
        text.set_text(f"Iteration: {frame + 1}")
        return pathline, point, text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        blit=True,
        interval=200,
        repeat=False,
    )

    plt.tight_layout()

    if show:
        plt.show()
    if file_path is not None:
        anim.save(file_path, writer="pillow", fps=fps)
    plt.close()


def load_metrics(file_path):
    with open(file_path, "rb") as file:
        train_loss, val_loss = pickle.load(file)
    return train_loss, val_loss
