import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional

matplotlib.use("TkAgg")  # fow WSL

Mesh = Tuple[np.ndarray, np.ndarray, np.ndarray]


class Plot:

    def __init__(self, mesh: Mesh, trajectory: Optional[np.ndarray] = None) -> None:
        self.mesh = mesh
        self.trajectory = trajectory

    @staticmethod
    def from_files(mesh_path: str) -> "Plot":

        trajectory = None
        with h5py.File(mesh_path, mode="r") as file:
            A = file["mesh"]["X"][:]
            B = file["mesh"]["Y"][:]
            Z = file["mesh"]["Z"][:]
            if "trajectory" in file["mesh"]:
                trajectory = file["mesh"]["trajectory"][:]
            assert A.shape == B.shape == Z.shape

        return Plot((A, B, Z), trajectory)

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
        self,
        levels: int = 10,
        plot_trajectory: bool = False,
        show: bool = True,
        file_path: Optional[str] = None,
    ) -> None:

        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(*self.mesh, cmap="viridis", levels=levels)
        ax.clabel(contour, inline=1, fontsize=8)

        if plot_trajectory:
            assert self.trajectory is not None
            x, y = self.trajectory
            ax.plot(x, y, marker=".")

        plt.xlabel("alpha")
        plt.ylabel("beta")
        plt.colorbar(contour, ax=ax, label="loss")

        if file_path is not None:
            plt.savefig(file_path)
        if show:
            plt.show()
        else:
            plt.close()
