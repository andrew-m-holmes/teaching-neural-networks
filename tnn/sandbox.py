import tnn
import numpy as np
from typing import Tuple


def f(x: float, y: float) -> float:
    global_min = np.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    local_min = 0.5 * np.exp(-((x + 1) ** 2 + (y + 1) ** 2))
    slopes = 0.1 * np.sin(3 * (x + y))
    return -(global_min + local_min + slopes) + 1


def df(x: float, y: float) -> Tuple[float, float]:
    global_min = np.exp(-((x - 1) ** 2 + (y - 1) ** 2))

    d_global_min_dx = -2 * (x - 1) * global_min
    d_local_min_dx = -0.5 * (x + 1) * np.exp(-((x + 1) ** 2 + (y + 1) ** 2))
    d_slopes_dx = 0.3 * np.cos(3 * (x + y))
    d_dx = -(d_global_min_dx + d_local_min_dx + d_slopes_dx)

    d_global_min_dy = -2 * (y - 1) * global_min
    d_local_min_dy = -0.5 * (y + 1) * np.exp(-((x + 1) ** 2 + (y + 1) ** 2))
    d_slopes_dy = 0.3 * np.cos(3 * (x + y))
    d_dy = -(d_global_min_dy + d_local_min_dy + d_slopes_dy)

    return (d_dx, d_dy)


def main():

    endpoints = (-2, 2)
    init = (-1.5, 0.9)
    lr = 5e-1
    iters = 40

    tnn.animate_function_descent_3d(
        f,
        df,
        endpoints,
        slack=0.1,
        init=init,
        lr=lr,
        verbose=10,
        iters=iters,
        path="../images/loss-anim-ravine.gif",
    )

    init = (-1.5, 1.1)

    tnn.animate_function_descent_3d(
        f,
        df,
        endpoints,
        slack=0.1,
        init=init,
        lr=lr,
        verbose=10,
        iters=iters,
        path="../images/loss-anim-momentum.gif",
    )


if __name__ == "__main__":
    main()
