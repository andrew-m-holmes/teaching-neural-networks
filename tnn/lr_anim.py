import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PATH = "../images"


# Parabola loss function and its derivative
def loss(x):
    return x**2


def grad(x):
    return 2 * x


def animate_learning_rate(learning_rate, ax, color):
    x = 10  # Initial parameter value
    xs, losses = [], []

    for _ in range(100):  # Simulate 100 steps
        xs.append(x)
        losses.append(loss(x))
        x -= learning_rate * grad(x)
        if abs(x) > 50:  # Stop if it diverges
            break

    # Plot the loss curve
    (line,) = ax.plot([], [], color=color, lw=2)

    def update(frame):
        line.set_data(xs[:frame], losses[:frame])
        return (line,)

    return update, len(xs)


def main():
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 100)

    # Learning rates
    learning_rates = [0.01, 0.1, 1.5, 0.3]
    colors = ["blue", "green", "red", "purple"]
    labels = ["Slow", "Good", "Diverging", "Oscillating"]

    animators = []
    for lr, color, label in zip(learning_rates, colors, labels):
        ax.set_title(f"Learning Rate: {label}")
        update_func, frames = animate_learning_rate(lr, ax, color)
        anim = FuncAnimation(fig, update_func, frames=frames, blit=True)
        animators.append(anim)

    for lr, anim in zip(learning_rates, animators):
        anim.save(f"{PATH}/learning-rate-{lr}.gif", writer="pillow", fps=5)


if __name__ == "__main__":
    main()
