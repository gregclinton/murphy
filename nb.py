import numpy as np

class Classifier:
    '''
    murphy pp. 84 to 89
    binary features only (this implementation)
    '''    
    def fit(self, X, y):        
        N, D = X.shape
        C = len(np.unique(y))
        self.priors = np.empty(C)
        self.thetas = np.empty((C, D))
        pseudocount = 1
        
        for c in range(C):
            i = y == c
            X_c = X[i]
            Non = np.sum(X_c == 1, axis = 0)
            Noff = np.sum(X_c == 0, axis = 0)
            self.priors[c] = np.mean(i)
            self.thetas[c, :] = 1.0 * (Non + pseudocount) / (Non + Noff + 2 * pseudocount)
        
    def predict(self, X):
        N, D = X.shape
        C = len(self.priors)
        eps = np.spacing(0)
        not_X = 1 - X
        log_priors = np.log(self.priors)
        log_theta = np.log(self.thetas + eps)
        log_not_theta = np.log(1 - self.thetas + eps)        
        log_post = np.empty((N, C))
        
        for c in range(C):
            L1 = X * log_theta[c, :]
            L0 = not_X * log_not_theta[c, :]
            log_post[:, c] = log_priors[c] + np.sum(L1 + L0, axis = 1)

        return np.argmax(log_post, axis = 1)