import numpy as np

class Classifier:
    def __init__(self, get_model, get_log_likelihood):
        self.get_model = get_model
        self.get_log_likelihood = get_log_likelihood
        
    def fit(self, X, y):        
        N, D = X.shape
        C = len(np.unique(y))
        self.thetas = self.get_model(X, y, N, D, C)
        self.priors = np.empty(C)
        
        for c in range(C):
            self.priors[c] = np.mean(y == c)
            
    def predict(self, X):
        N, D = X.shape
        C = len(self.thetas)
        log_likelihood = self.get_log_likelihood(X, N, D, C, self.thetas)

        log_priors = np.log(self.priors)
        log_post = log_likelihood
        for c in range(C):
            log_post[:, c] += log_priors[c]
            
        return np.argmax(log_post, axis = 1)