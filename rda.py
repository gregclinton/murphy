import numpy as np
import math
from numpy.linalg import inv, svd
import generative


class Classifier(generative.Classifier):
    '''
    regularized discriminant analysis classifier
    murphy p. 107
    '''        
    def __init__(self, lmbda = 0):
        self.lmbda = 0.0
        
        def get_theta(X, y, N, D, C):
      
            U, s, V = svd(X, full_matrices = False)
            V = V.T
            Z = U.dot(np.diag(s))
            Sigma = np.cov(Z, rowvar = False, ddof = 0)
            Sigma = self.lmbda * np.diag(Sigma) + (1 - lmbda) * Sigma
            VInvSigma = V.dot(inv(Sigma))
            Betas = []

            for c in range(C):
                X_c = X[y == c]
                mu_c = np.mean(X_c, axis = 0)
                mu_z_c = V.T.dot(mu_c)
                Beta = VInvSigma.dot(mu_z_c)
                gamma = -mu_c.T.dot(Beta) / 2.0
                Betas.append(np.hstack((Beta, gamma)))

            return np.transpose(Betas)

        def fill_log_likelihood(X, N, D, C, theta, log_likelihood):
            Beta = theta

            for c in range(C):
                log_likelihood[:, c] = X.dot(Beta[c])
        
        generative.Classifier.__init__(self, get_theta, fill_log_likelihood)