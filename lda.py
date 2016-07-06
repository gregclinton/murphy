import numpy as np
from numpy.linalg import inv

class LDA:        
    def fit(self, X, y):
        self.c = np.unique(y)
        Sigma = np.cov(X, rowvar = False)
        InvSigma = inv(Sigma)
        
        def theta(c):
            prior = 1.0 * sum(y == c) / len(y)
            mean = np.mean(X[y == c], axis = 0)
            gamma = -mean.dot(InvSigma).dot(mean) / 2.0 + np.log(prior)
            Beta = InvSigma.dot(mean)
            return (gamma, Beta)
            
        self.theta = [theta(c) for c in self.c]

    def predict(self, X):
        p = []
        
        for theta in self.theta:
            gamma, Beta = theta
            p.append(X.dot(Beta) + gamma)
        return np.array([self.c[i] for i in np.argmax(p, axis = 0)])