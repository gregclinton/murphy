import numpy as np

class Classifier:
    def __init__(self, k = 10):
        self.k = k
        
    def fit(self, X, y):
        self.theta = X, y

    def predict(self, XX):
        X, y = self.theta
        aTa = lambda a: np.dot(a.T, a)
        dist = lambda x, y: aTa(x - y)
        def pred(x):
            closest = np.argsort([dist(x, xx) for xx in X])[0 : self.k]
            return y[np.argmax(np.bincount(closest))]
        return np.array([pred(x) for x in XX])