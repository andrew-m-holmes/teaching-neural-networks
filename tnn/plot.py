import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from typing import Dict, Optional, List, Tuple, Callable

plt.style.use("dark_background")


def plot_metrics(
    metrics: Dict[str, np.ndarray],
    best_linestyle: str = "-",
    use_min: bool = True,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: str = "epoch",
    ylabel: str = "loss",
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
    path: Optional[str] = None,
) -> None:
    plt.figure(figsize=figsize)

    if colors is None:
        colors = ["dodgerblue", "red", "blueviolet", "aquamarine", "coral", "purple"]

    default_linestyle = "-"
    func = min if use_min else max
    best_metric = func(metrics, key=lambda item: item[1][-1])

    for (metric_name, values), color in zip(metrics.items(), colors):
        linestyle = best_linestyle if metric_name == best_metric else default_linestyle
        epochs = range(1, len(values) + 1)
        final_value = values[-1]
        plt.plot(
            epochs,
            values,
            label=f"{metric_name}: {final_value:.4f}",
            color=color,
            linestyle=linestyle,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(loc="best")
    plt.style.use("dark_background")

    if path is not None:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def plot_surface_3D(
    meshgrid: np.ndarray,
    cmap: str = "viridis",
    show: bool = True,
    path: Optional[str] = None,
) -> None:

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.grid(False)

    ax.plot_surface(*meshgrid, cmap=cmap, alpha=0.5)

    ax.set_xlabel("x direction", color="gray")
    ax.set_ylabel("y direction", color="gray")
    ax.set_zlabel("loss", color="gray")

    ax.tick_params(axis="x", colors="gray")
    ax.tick_params(axis="y", colors="gray")
    ax.tick_params(axis="z", colors="gray")

    if path is not None:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_contour(
    meshgrid: np.ndarray,
    optim_path: Optional[np.ndarray] = None,
    variance: Optional[np.ndarray] = None,
    levels: int = 100,
    cmap: str = "viridis",
    show: bool = True,
    path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    contour = ax.contourf(*meshgrid, cmap=cmap, levels=levels, antialiased=True)
    if optim_path is not None:
        x, y = optim_path.T
        ax.plot(x, y, marker=".", color="dodgerblue", linewidth=2, markersize=8)

    if variance is not None:
        var_1, var_2 = variance
        ax.set_xlabel(f"principal component 1: {var_1:.3f}", color="white")
        ax.set_ylabel(f"principal component 2: {var_2:.3f}", color="white")
    else:
        ax.set_xlabel("principal component 1", color="white")
        ax.set_ylabel("principal component 2", color="white")

    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("loss", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    plt.tight_layout()

    if path is not None:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        plt.savefig(path, facecolor="black", edgecolor="none")
    if show:
        plt.show()
    else:
        plt.close()


def animate_contour(
    meshgrid: np.ndarray,
    optim_path: np.ndarray,
    variance: Optional[np.ndarray] = None,
    fps: int = 5,
    levels: int = 100,
    cmap: str = "viridis",
    show: bool = True,
    path: Optional[str] = None,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    contour = ax.contourf(*meshgrid, cmap=cmap, levels=levels, antialiased=True)

    ax.set_xticks([])
    ax.set_yticks([])

    if variance is not None:
        var_1, var_2 = variance
        ax.set_xlabel(f"principle component 1: {var_1:.3f}", color="white")
        ax.set_ylabel(f"principle component 2: {var_2:.3f}", color="white")
    else:
        ax.set_xlabel("principle component 1", color="white")
        ax.set_ylabel("principle component 2", color="white")

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("loss", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    pc_0 = optim_path[0]
    pcx, pcy = [pc_0[0]], [pc_0[1]]
    (pathline,) = ax.plot(pcx, pcy, color="dodgerblue", lw=2)
    (point,) = ax.plot(pcx, pcy, "o", color="dodgerblue", markersize=8)
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

    def update(frame):
        pc = optim_path[frame]
        pcx.append(pc[0])
        pcy.append(pc[1])
        pathline.set_data(pcx, pcy)
        point.set_data([pcx[-1]], [pcy[-1]])
        text.set_text(f"iteration: {frame + 1}")
        return pathline, point, text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(optim_path),
        blit=True,
        interval=200,
        repeat=False,
    )
    plt.tight_layout()

    if show:
        plt.show()
    if path is not None:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        anim.save(path, writer="pillow", fps=fps)
    plt.close()


def animate_function_descent_3d(
    func: Callable[..., float],
    dfunc: Callable[..., Tuple[float, float]],
    init: Optional[Tuple[float, float]] = None,
    fps: int = 5,
    cmap: str = "viridis",
    show: bool = True,
    path: Optional[str] = None,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))

    def update():
        pass

    anim = FuncAnimation(
        fig=fig,
        func=update,
    )

    if show:
        pass
    if path is not None:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
