import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class polymod:

    def __init__(self, ndim: int) -> None:
        self.coefficients = np.random.randint(-1, 1, ndim).astype(
            float
        ) * np.random.randn(ndim)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.sum(input * self.coefficients, axis=1)


def main():

    np.random.seed(0)
    ndim = 5
    data = np.random.randn(100, ndim)

    pca = PCA(n_components=2, svd_solver="covariance_eigh")
    model = polymod(ndim)
    pca.fit(model.coefficients.reshape(1, -1))
