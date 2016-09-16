import numpy as np
import generative

class Classifier(generative.Classifier):
    '''
    naive bayes classifier (features treated as independent given theta)
    murphy pp. 84 to 89
    binary features only (this implementation)
    '''        
    def __init__(self):
        def get_theta(X, y, N, D, C):
            theta = np.empty((C, D))
            pseudocount = 1

            for c in range(C):
                X_c = X[y == c]
                Non = np.sum(X_c == 1, axis = 0)
                Noff = np.sum(X_c == 0, axis = 0)
                theta[c, :] = 1.0 * (Non + pseudocount) / (Non + Noff + 2 * pseudocount)

            return theta

        def get_log_likelihood(X, N, D, C, theta):
            eps = np.spacing(0)
            not_X = 1 - X
            log_theta = np.log(theta + eps)
            log_not_theta = np.log(1 - theta + eps)        
            log_likelihood = np.empty((N, C))

            for c in range(C):
                L1 = X * log_theta[c, :]
                L0 = not_X * log_not_theta[c, :]
                log_likelihood[:, c] = np.sum(L1 + L0, axis = 1)

            return log_likelihood
        
        generative.Classifier.__init__(self, get_theta, get_log_likelihood)