import numpy as np
import generative

class Classifier(generative.Classifier):
    '''
    naive bayes classifier (features treated as independent given theta)
    a generative classifier
    murphy pp. 84 to 89
    binary features only (this implementation)
    ccd is class conditional density
    '''
    
    def log_ccd(theta):
        pass
        
    def __init__(self):
        def get_model(X, y, N, D, C):
            thetas = np.empty((C, D))
            pseudocount = 1

            for c in range(C):
                X_c = X[y == c]
                Non = np.sum(X_c == 1, axis = 0)
                Noff = np.sum(X_c == 0, axis = 0)
                thetas[c, :] = 1.0 * (Non + pseudocount) / (Non + Noff + 2 * pseudocount)

            return thetas

        def get_log_likelihood(X, N, D, C, thetas):
            eps = np.spacing(0)
            not_X = 1 - X
            log_theta = np.log(thetas + eps)
            log_not_theta = np.log(1 - thetas + eps)        
            log_likelihood = np.empty((N, C))

            for c in range(C):
                L1 = X * log_theta[c, :]
                L0 = not_X * log_not_theta[c, :]
                log_likelihood[:, c] = np.sum(L1 + L0, axis = 1)

            return log_likelihood
        
        generative.Classifier.__init__(self, get_model, get_log_likelihood)