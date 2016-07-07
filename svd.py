import numpy as np
from numpy import svd

X = np.array([
    [2, 3, 4, 5],
    [2.2, 3.1, 4.2, 5.3],
    [2.1, 3.2, 4.1, 5.2],
])

U, s, V = svd(X, full_matrices = False)
D = np.diag(s)
V = V.T
print U.dot(D).dot(V.T)