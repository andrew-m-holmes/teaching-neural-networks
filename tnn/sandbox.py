import tnn
import numpy as np
from typing import Tuple


def f(x: np.ndarray, w1: float, w2: float) -> float:
    return (abs(x) * (w1**2 + w2**2)).mean()


def df(x: np.ndarray, w1: float, w2: float, grad: float) -> Tuple[float, float]:
    dw1 = grad * (2 * w1 * abs(x))
    dw2 = grad * (2 * w2 * abs(x))
    return dw1.sum(), dw2.sum()


def main():

    n_features = 16
    endpoints = (-1, 1)
    init = (0.983, -0.821)
    lr = 75e-4
    iters = 20

    tnn.animate_function_descent_3d(
        f,
        df,
        endpoints,
        n_features,
        slack=0.1,
        init=init,
        lr=lr,
        verbose=True,
        iters=iters,
        path="../images/loss-anim.gif",
    )


if __name__ == "__main__":
    main()
