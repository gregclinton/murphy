import numpy as np
from scipy import stats

class QDA:        
    def fit(self, X, y):
        self.c = np.unique(y)
        
        def theta(c):
            X_c = X[y == c]
            prior = 1.0 * sum(y == c) / len(y)
            mean = np.mean(X_c, axis = 0)
            Sigma = np.cov(X_c, rowvar = False)
            return prior, mean, Sigma
            
        self.theta = [theta(c) for c in self.c]

    def predict(self, X):
        p = []
        
        for theta in self.theta:
            prior, mean, Sigma = theta
            p.append(prior * stats.multivariate_normal.pdf(X, mean, Sigma))
            
        return np.array([self.c[i] for i in np.argmax(p, axis = 0)])
    
# from sklearn import qda
# clf = qda.QDA()

def demo():
    X = np.array([[67, 21], [79, 23], [71, 22.2], [68, 20], [67, 21], [60, 21.5]])
    y = np.array([1, 1, 1, 2, 2, 2])
    clf = QDA()
    clf.fit(X, y)
    print clf.predict([[66, 21], [70, 23], [70, 23]])