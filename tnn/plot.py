import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from typing import Dict, Optional, List, Tuple, Callable, Union

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

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if title:
        plt.title(title)
    plt.legend(loc="best", fontsize=14)
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
    endpoints: Tuple[float, float],
    n_features: int,
    slack: float = 0.5,
    init: Optional[Tuple[float, float]] = None,
    lr: float = 1e-2,
    grad: float = 1.0,
    iters: int = 100,
    repeat: bool = False,
    fps: int = 5,
    cmap: str = "viridis",
    show: bool = True,
    path: Optional[str] = None,
    verbose: Optional[Union[int, bool]] = None,
) -> None:

    if not verbose or verbose < 0:
        verbose = False
    else:
        verbose = int(verbose)

    n_points = 100
    x_coords = np.linspace(*endpoints, num=n_points)
    y_coords = np.linspace(*endpoints, num=n_points)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.zeros((n_points, n_points))

    x = np.random.randn(n_features)
    for i in range(n_points):
        for j in range(n_points):
            w1, w2 = X[i, j], Y[i, j]
            loss = func(x, w1, w2)
            Z[i, j] = loss

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    param_range = (min(endpoints) + slack, max(endpoints))
    loss_range = (0, Z.max() + slack)

    plt.xlabel("w1")
    plt.ylabel("w2")
    ax.set_zlabel("loss")
    ax.set_xlim(param_range)
    ax.set_ylim(param_range)
    ax.set_zlim(loss_range)

    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.5)
    scatter = ax.scatter([], [], [], color="red", s=25, depthshade=False)
    w1, w2 = (
        init if init is not None else np.random.choice(x_coords, size=2, replace=True)
    )

    def update(frame):
        nonlocal w1, w2
        loss = func(x, w1, w2)
        scatter._offsets3d = ([w1], [w2], [loss])
        dw1, dw2 = dfunc(x, w1, w2, grad)
        w1 -= lr * dw1
        w2 -= lr * dw2
        if ((frame + 1) % verbose == 0 and frame) or (frame + 1) == verbose:
            print(f"frame: {frame + 1}, loss: {loss:.3f}, w1: {w1:.3f}, w2: {w2:.3f}")
        return (scatter,)

    anim = FuncAnimation(
        fig=fig, func=update, frames=iters, interval=200, blit=True, repeat=repeat
    )

    if show and path is None:
        plt.show()
    if path is not None:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        anim.save(f"{path}", writer="pillow", fps=fps)
