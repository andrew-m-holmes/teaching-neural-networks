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
    best_metric = func(metrics.items(), key=lambda item: item[1][-1])[0]
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


def animate_function_descent_3d(
    func: Callable[..., float],
    dfunc: Callable[..., Tuple[float, float]],
    start: float,
    stop: float,
    step: int,
    use_logspace: bool = False,
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

    if use_logspace:
        logspace = np.logspace(start, stop, num=step)
        coords = np.concat((np.negative(logspace)[::-1], [0], logspace))
    else:
        coords = np.linspace(start, stop, num=step)

    points = coords.size
    X, Y = np.meshgrid(coords, coords)
    Z = np.zeros((points, points))

    for i in range(points):
        for j in range(points):
            w1, w2 = X[i, j], Y[i, j]
            loss = func(w1, w2)
            Z[i, j] = loss

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    param_range = (min(coords) + slack, max(coords))
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
        init if init is not None else np.random.choice(coords, size=2, replace=True)
    )

    def update(frame):
        nonlocal w1, w2
        loss = func(w1, w2)
        scatter._offsets3d = ([w1], [w2], [loss])
        dw1, dw2 = dfunc(w1, w2)
        w1 -= lr * dw1 * grad
        w2 -= lr * dw2 * grad
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
