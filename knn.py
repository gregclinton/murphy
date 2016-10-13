import numpy as np

class Classifier:
    def __init__(self, k = 10):
        self.k = k
        
    def fit(self, X, y):
        self.theta = X, y

    def predict(self, XX):
        X, y = self.theta
        aTa = lambda a: np.dot(a, a)
        dist = lambda x, y: aTa(x - y)
        mode = lambda a: np.argmax(np.bincount(a.astype(int)))
        pred = lambda x: mode(y[np.argsort([dist(x, xx) for xx in X])[0 : self.k]])
        return np.array([pred(x) for x in XX])