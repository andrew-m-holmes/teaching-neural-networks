import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x, y):
    return x**2 + y**2


def main():

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    xv, yv = np.meshgrid(x, y)
    z = f(xv, yv)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="cool")
    plt.show()


if __name__ == "__main__":
    main()
