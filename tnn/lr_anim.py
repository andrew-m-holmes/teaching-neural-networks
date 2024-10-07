import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_gradient_descent(learning_rate, start_point, num_steps):
    # Set up the plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Loss")
    ax.set_title(f"Gradient Descent Animation (Learning Rate: {learning_rate})")
    x = np.linspace(-6, 6, 100)
    y = x**2
    ax.plot(x, y, "b-", lw=2)

    # Initialize the point
    scatter = ax.scatter([], [], color="red")

    # Variables for gradient descent
    parameter = start_point

    # Update function for animation
    def update(frame):
        nonlocal parameter
        scatter._offset2d = ([parameter], [parameter**2])
        gradient = 2 * parameter
        parameter = parameter - learning_rate * gradient
        return (scatter,)

    anim = FuncAnimation(fig, update, frames=num_steps, interval=100, blit=True)

    # Save the animation
    anim.save(f"learning-rate-{learning_rate}.gif", writer="pillow", fps=30)
    plt.close(fig)


# Example usage
animate_gradient_descent(learning_rate=0.1, start_point=5, num_steps=50)
animate_gradient_descent(learning_rate=0.01, start_point=5, num_steps=50)
animate_gradient_descent(learning_rate=0.5, start_point=5, num_steps=50)
