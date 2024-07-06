import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple

matplotlib.use("TkAgg")  # fow WSL


def f(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a**2 * x + b**2 * x


def df(x: np.ndarray, a: float, b: float, grad: float) -> Tuple[float, float]:
    da = 2 * a * x * grad
    db = 2 * b * x * grad
    return da.sum(), db.sum()


def main():

    size = 100
    features = 16

    A = np.linspace(-1, 1, size)
    B = np.linspace(-1, 1, size)
    x = abs(np.random.randn(features))

    a, b = np.meshgrid(A, B)
    z = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            loss = f(x, a[i, j], b[i, j]).mean()
            z[i, j] = loss

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    param_range = (-1.5, 1.5)
    loss_range = (z.min() - 0.5, z.max() + 0.5)

    plt.xlabel("a")
    plt.ylabel("b")
    ax.set_zlabel("loss")
    ax.set_xlim(param_range)
    ax.set_ylim(param_range)
    ax.set_zlim(loss_range)
    ax.plot_surface(a, b, z, cmap="viridis", alpha=0.5)
    scatter = ax.scatter([], [], [], color="black", s=25, depthshade=False)

    iters = 10
    lr = 0.0075
    a, b = -1.33, 1.25
    grad = 1.0
    scheduler = 1

    def update(frame):
        nonlocal a, b
        a, b = float(a), float(b)
        loss = f(x, a, b).mean()
        scatter._offsets3d = ([a], [b], [loss])
        da, db = df(x, a, b, grad)
        a -= lr * da
        b -= lr * db
        if not ((frame + 1) % scheduler) or (frame + 1) == scheduler:
            print(f"Iteration: {frame + 1}, a: {a:.3f}, b: {b:.3f}, loss: {loss:.4f}")
        return (scatter,)

    path = os.path.abspath(os.path.dirname(__file__))
    anim = FuncAnimation(
        fig, update, frames=iters, interval=200, blit=True, repeat=False
    )
    anim.save(f"{path}/../images/grad-descent-anim.gif", writer="pillow", fps=5)


if __name__ == "__main__":
    main()
