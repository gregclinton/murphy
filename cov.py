import numpy as np

# np.cov(X, rowvar = False, ddof = 0)

def cov(X):
    mean = np.mean(X, axis = 0).reshape(-1, 1)
    return X.T.dot(X) / (len(X) * 1.0) - mean.dot(mean.T)

def cov(X):
    N, D = X.shape
    mean = np.mean(X, axis = 0).reshape(-1, 1)
    Sigma = np.zeros((D, D))
    
    for x in X:
        Sigma += (x.reshape(-1, 1) - mean) ** 2
        
    return Sigma / (N * 1.0)