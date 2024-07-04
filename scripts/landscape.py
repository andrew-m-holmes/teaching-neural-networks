import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.sqrt(a * x**2 + b * y**2)


def main():

    xls = np.linspace(-5, 5, 100)
    yls = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(xls, yls)
    a, b = 1.0, 1.0
    z = f(x, y, a, b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")
    plt.show()


if __name__ == "__main__":
    main()
