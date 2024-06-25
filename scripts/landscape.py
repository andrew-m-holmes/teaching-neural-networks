import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def polynomial(coefficients, x):
    return sum(c * (x**i) for i, c in enumerate(coefficients))


num_coefficients = 5
num_samples = 1000
coefficients = np.random.uniform(-10, 10, (num_samples, num_coefficients))
pca = PCA(n_components=2)

x = np.linspace(-1, 1, 100)
loss = np.array([np.mean(polynomial(coeff, x) ** 2) for coeff in coefficients])
