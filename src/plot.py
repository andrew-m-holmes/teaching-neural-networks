import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional

matplotlib.use("TkAgg")  # fow WSL

LossLandscape = Tuple[np.ndarray, np.ndarray, np.ndarray]
Trajectory = List[np.ndarray]


class Plot:

    def __init__(
        self, losslandscape: LossLandscape, trajectory: Optional[Trajectory]
    ) -> None:
        self.losslandscape = losslandscape
        self.trajectory = trajectory

    @staticmethod
    def fromfiles(llspath: str, trajpath: Optional[str] = None) -> None:
        with h5py.File(llspath, mode="r") as file:
            A = file["axes"]["A"][:]
            B = file["axes"]["B"][:]
            Z = file["axes"]["Z"][:]
            assert A.shape == B.shape == Z.shape

        if trajpath is not None:
            with h5py.File(trajpath, mode="r") as file:
                pass

        return Plot((A, B, Z), None)

    def plotsurface3D(self, show: bool = True, filepath: Optional[str] = None) -> None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlabel("alpha")
        plt.ylabel("beta")
        ax.set_zlabel("loss")
        ax.plot_surface(*self.losslandscape, cmap="viridis", alpha=0.5)

        if filepath is not None:
            plt.savefig(filepath)
        if show:
            plt.show()
        else:
            plt.close()
