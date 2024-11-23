import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from typing import Optional

plt.style.use("dark_background")


class GradientDescentVisualizer:
    def __init__(
        self,
        learning_rate: float,
        x_start: float,
        n_iterations: int,
        show: bool = False,
        path: Optional[str] = None,
        fps: int = 5,
    ):
        self.lr = learning_rate
        self.x_start = x_start
        self.n_iterations = n_iterations
        self.show = show
        self.path = path
        self.fps = fps
        self.trajectory = []
        self.current_x = x_start
        self._calculate_trajectory()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def loss_function(self, x):
        return x**2

    def gradient(self, x):
        return 2 * x

    def _calculate_trajectory(self):
        self.trajectory = [self.x_start]
        current_x = self.x_start

        for _ in range(self.n_iterations):
            grad = self.gradient(current_x)
            current_x = current_x - self.lr * grad
            self.trajectory.append(current_x)

    def _init_animation(self):
        self.ax.clear()

        x = np.linspace(-5, 5, 200)
        y = self.loss_function(x)
        self.ax.plot(x, y, "-", label="Loss Function: f(x) = x²", color="coral")

        (self.point,) = self.ax.plot(
            [], [], "o", markersize=10, label="Current Position", color="dodgerblue"
        )

        (self.line,) = self.ax.plot(
            [], [], "--", alpha=0.5, label="Trajectory", color="red"
        )

        self.iteration_text = self.ax.text(
            0.02,
            0.95,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor="black",
                edgecolor="white",
                alpha=0.7,
                boxstyle="round,pad=0.5",
            ),
        )

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-1, 30)
        self.ax.grid(False)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        return self.point, self.line

    def _animate(self, frame):
        self.line.set_data(
            self.trajectory[: frame + 1],
            [self.loss_function(x) for x in self.trajectory[: frame + 1]],
        )

        current_x = self.trajectory[frame]
        self.point.set_data([current_x], [self.loss_function(current_x)])
        self.iteration_text.set_text(f"Iteration: {frame}")

        return self.point, self.line

    def create_animation(self):
        anim = FuncAnimation(
            self.fig,
            self._animate,
            init_func=self._init_animation,
            frames=len(self.trajectory),
            interval=100,
            blit=True,
        )
        if self.show:
            plt.show()
        if self.path is not None:
            dirname = os.path.dirname(self.path)
            os.makedirs(dirname, exist_ok=True)
            anim.save(f"{self.path}", writer="pillow", fps=self.fps)


def visualize_gradient_descent(
    learning_rate: float,
    x_start: float,
    iters: int,
    show: bool = False,
    path: Optional[str] = None,
):
    visualizer = GradientDescentVisualizer(
        learning_rate=learning_rate,
        x_start=x_start,
        n_iterations=iters,
        show=show,
        path=path,
    )
    visualizer.create_animation()


def main():
    learning_rates = {
        "converging": (5, 0.9),
        "slow": (5, 0.001),
        "oscilating": (5, 0.999),
        "diverging": (3, 1.09),
    }
    for convergence, (x_start, lr) in learning_rates.items():
        visualize_gradient_descent(
            learning_rate=lr,
            x_start=x_start,
            iters=20,
            path=f"../images/lr-descent-{convergence}.gif",
        )


if __name__ == "__main__":
    main()
