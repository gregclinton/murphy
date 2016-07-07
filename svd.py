U, s, V = svd(X, full_matrices = False)
D = np.diag(s)
V = V.T
print U.dot(D).dot(V.T)