import numpy as np
from numpy.linalg import inv
import generative

class Classifier(generative.Classifier):
    '''
    linear discriminant analysis classifier
    murphy p. 105
    or use LDA from sklearn.lda
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

        def fill_log_likelihood(X, N, D, C, theta, log_likelihood):
            Betas, gammas = theta

            for c in range(C):
                log_likelihood[:, c] = X.dot(Betas[i]) + gammas[c]
        
        generative.Classifier.__init__(self, get_theta, fill_log_likelihood)