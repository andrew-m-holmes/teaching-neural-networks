import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional

matplotlib.use("TkAgg")  # fow WSL

Mesh = Tuple[np.ndarray, np.ndarray, np.ndarray]


class Plot:

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    @staticmethod
    def from_files(mesh_path: str) -> "Plot":
        with h5py.File(mesh_path, mode="r") as file:
            A = file["mesh"]["X"][:]
            B = file["mesh"]["Y"][:]
            Z = file["mesh"]["Z"][:]
            assert A.shape == B.shape == Z.shape

        return Plot((A, B, Z))

    def plot_surface_3D(
        self, show: bool = True, file_path: Optional[str] = None
    ) -> None:

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlabel("alpha")
        plt.ylabel("beta")
        ax.set_zlabel("loss")
        ax.plot_surface(*self.mesh, cmap="viridis", alpha=0.5)

        if file_path is not None:
            plt.savefig(file_path)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_contour(
        self, levels: int = 20, show: bool = True, file_path: Optional[str] = None
    ) -> None:

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.xlabel("alpha")
        plt.ylabel("beta")
        contour = ax.contourf(*self.mesh, cmap="viridis", levels=levels)

        plt.colorbar(contour, ax=ax, label="loss")

        if file_path is not None:
            plt.savefig(file_path)
        if show:
            plt.show()
        else:
            plt.close()
