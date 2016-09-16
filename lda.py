import numpy as np
from numpy.linalg import inv
import generative

class Classifier(generative.Classifier):
    '''
    linear discriminant analysis classifier
    murphy p. 105
    or from sklearn import lda
    '''        
    def __init__(self):
        def get_theta(X, y, N, D, C):
            gammas = []
            Betas = []
            Sigma = np.cov(X, rowvar = False)
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