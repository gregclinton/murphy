import numpy as np

class Classifier:
    '''
    naive bayes classifier
    murphy pp. 84 to 89
    binary features only (this implementation)
    '''
    def nb_fit(self, X, y, N, D, C):
        self.priors = np.empty(C)
        self.thetas = np.empty((C, D))
        pseudocount = 1
        
        for c in range(C):
            X_c = X[y == c]
            Non = np.sum(X_c == 1, axis = 0)
            Noff = np.sum(X_c == 0, axis = 0)
            self.thetas[c, :] = 1.0 * (Non + pseudocount) / (Non + Noff + 2 * pseudocount)

    def nb_predict(self, X, N, D, C, thetas):
        eps = np.spacing(0)
        not_X = 1 - X
        log_theta = np.log(self.thetas + eps)
        log_not_theta = np.log(1 - self.thetas + eps)        
        log_likelihood = np.empty((N, C))
        
        for c in range(C):
            L1 = X * log_theta[c, :]
            L0 = not_X * log_not_theta[c, :]
            log_likelihood[:, c] = np.sum(L1 + L0, axis = 1)
            
        return log_likelihood
    
    def fit(self, X, y):        
        N, D = X.shape
        C = len(np.unique(y))
        self.nb_fit(X, y, N, D, C)
        
        for c in range(C):
            self.priors[c] = np.mean(y == c)
            
    def predict(self, X):
        N, D = X.shape
        C = len(self.thetas)
        log_likelihood = self.nb_predict(X, N, D, C, self.thetas)

        log_priors = np.log(self.priors)
        log_post = log_likelihood
        for c in range(C):
            log_post[:, c] += log_priors[c]
            
        return np.argmax(log_post, axis = 1)