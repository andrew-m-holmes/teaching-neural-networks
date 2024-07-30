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
        self, cmap: str = "viridis", show: bool = True, file_path: Optional[str] = None
    ) -> None:

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.grid(False)

        ax.plot_surface(*self.mesh, cmap=cmap, alpha=0.5)

        ax.set_xlabel("Principle Component 1", color="gray")
        ax.set_ylabel("Principle Component 2", color="gray")
        ax.set_zlabel("Loss", color="gray")

        ax.tick_params(axis="x", colors="gray")
        ax.tick_params(axis="y", colors="gray")
        ax.tick_params(axis="z", colors="gray")

        if file_path is not None:
            plt.savefig(file_path)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_contour(
        self,
        levels: int = 100,
        plot_trajectory: bool = False,
        cmap: str = "viridis",
        show: bool = True,
        file_path: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        contour = ax.contourf(*self.mesh, cmap=cmap, levels=levels, antialiased=True)
        if plot_trajectory:
            assert self.trajectory is not None
            x, y = self.trajectory
            ax.plot(x, y, marker=".", color="dodgerblue", linewidth=2, markersize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Principal Component 1", color="white")
        ax.set_ylabel("Principal Component 2", color="white")
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Loss", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        plt.tight_layout()

        if file_path is not None:
            plt.savefig(file_path, facecolor="black", edgecolor="none")
        if show:
            plt.show()
        else:
            plt.close()

    def animate_contour(
        self,
        levels: int = 100,
        fps: int = 5,
        cmap: str = "viridis",
        show: bool = True,
        file_path: Optional[str] = None,
    ) -> None:

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        contour = ax.contourf(*self.mesh, cmap=cmap, levels=levels, antialiased=True)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Principal Component 1", color="white")
        ax.set_ylabel("Principal Component 2", color="white")

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Loss", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        assert self.trajectory is not None
        trajectory = self.trajectory.T
        pc_0 = trajectory[0]
        pcx, pcy = [pc_0[0]], [pc_0[1]]
        (pathline,) = ax.plot(pcx, pcy, color="dodgerblue", lw=2)
        (point,) = ax.plot(pcx, pcy, "o", color="dodgerblue", markersize=8)
        text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

        def update(frame):
            pc = trajectory[frame]
            pcx.append(pc[0])
            pcy.append(pc[1])
            pathline.set_data(pcx, pcy)
            point.set_data([pcx[-1]], [pcy[-1]])
            text.set_text(f"Iteration: {frame + 1}")
            return pathline, point, text

        anim = FuncAnimation(
            fig,
            update,
            frames=len(trajectory),
            blit=True,
            interval=200,
            repeat=False,
        )

        plt.tight_layout()

        if show:
            plt.show()
        if file_path is not None:
            anim.save(file_path, writer="pillow", fps=fps)
        plt.close()
