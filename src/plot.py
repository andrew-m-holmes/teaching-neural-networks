import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple

matplotlib.use("TkAgg")  # fow WSL


def plotsurface3D(filepath: str) -> None:
    with h5py.File(filepath, mode="r") as file:
        A = file["axes"]["A"]
        B = file["axes"]["B"]
        Z = file["axes"]["Z"]

    assert A.shape == B.shape == Z.shape
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    ax.set_zlabel("loss")
    ax.plot_surface(A, B, Z, cmap="viridis", alpha=0.5)
    plt.show()
