import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional

# matplotlib.use("TkAgg")  # fow WSL

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
        levels: int = 35,
        plot_trajectory: bool = False,
        cmap: str = "viridis",
        show: bool = True,
        file_path: Optional[str] = None,
    ) -> None:

        _, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(*self.mesh, cmap=cmap, levels=levels)
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

    def animate_contour(
        self,
        levels: int = 35,
        fps: int = 5,
        cmap: str = "viridis",
        file_path: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(*self.mesh, cmap=cmap, levels=levels)
        ax.clabel(contour, inline=1, fontsize=8)

        plt.xlabel("alpha")
        plt.ylabel("beta")
        plt.colorbar(contour, ax=ax, label="loss")

        assert self.trajectory is not None
        trajectory = self.trajectory.T

        pc_0 = trajectory[0]
        pcx, pcy = [pc_0[0]], [pc_0[1]]
        (pathline,) = ax.plot(pcx, pcy, color="blue", lw=1)
        (point,) = ax.plot(pcx, pcy, "ro", markersize=10)
        text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        def update(frame):

            pc = trajectory[frame]
            pcx.append(pc[0])
            pcy.append(pc[1])
            pathline.set_data(pcx, pcy)
            point.set_data([pcx[-1]], [pcy[-1]])
            text.set_text(f"Iteration: {frame + 1}")
            return point, text

        global anim
        anim = FuncAnimation(
            fig,
            update,
            frames=len(trajectory),
            blit=True,
            interval=200,
            repeat=False,
        )

        if file_path is not None:
            anim.save(file_path, writer="pillow", fps=fps)
        plt.close()
