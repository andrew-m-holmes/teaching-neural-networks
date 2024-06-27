import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def polynomial(coefficients, x):
    return sum(c * (x**i) for i, c in enumerate(coefficients))
