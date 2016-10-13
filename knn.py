import numpy as np

aTa = lambda a: np.dot(a, a)
dist = lambda x, y: aTa(x - y)
mode = lambda a: np.argmax(np.bincount(a))

def train(X, y):
    return X, y
                       
def predict(model, x, k, dist = dist):
    X, y = model
    nearest = np.argsort([dist(x, xx) for xx in X])[0 : k]
    return mode(y[nearest])