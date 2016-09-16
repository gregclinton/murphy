import numpy as np
from numpy.linalg import inv
import generative

'''
naive bayes classifiers (features treated as independent given theta)
murphy pp. 84 to 89
'''        

class Bernoulli(generative.Classifier):
    '''
    binary features only (this implementation)
    from sklearn.naive_bayes import BernoulliNB
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


class Gaussian(generative.Classifier):
    '''
    from sklearn.naive_bayes import GaussianNB
    '''        
    def __init__(self):
        def get_theta(X, y, N, D, C):
            gammas = []
            Betas = []
            Sigma = np.cov(X, rowvar = False)
            Sigma = np.diag(np.diag(Sigma))
            InvSigma = inv(Sigma)

            for c in range(C):
                mean = np.mean(X[y == c], axis = 0)
                gammas.append(-mean.dot(InvSigma).dot(mean) / 2.0)
                Betas.append(InvSigma.dot(mean))
        
            return Betas, gammas

        def get_log_likelihood(X, N, D, C, theta):
            Betas, gammas = theta
            log_likelihood = np.empty((N, C))

            for i in range(C):
                log_likelihood[:, i] = X.dot(Betas[i]) + gammas[i]
            return log_likelihood
        
        generative.Classifier.__init__(self, get_theta, get_log_likelihood)        