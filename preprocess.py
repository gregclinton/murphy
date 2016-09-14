import numpy as np

def scale(X):
    # see sklearn.preprocessing.scale
    mean = np.mean(X, axis = 0)
    cov = np.cov(X, ddof = 0, rowvar = False)
    return (X - mean) / np.sqrt(np.diag(cov))