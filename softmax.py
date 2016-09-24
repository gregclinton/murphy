import numpy as np

def works(x):
    x = np.array(x, dtype = float)
    x = x.T - np.max(x, axis = 1) # prevent overflow
    e = np.exp(x)
    return x, e, np.sum(e, axis = 0)

def softmax(x):
    _, e, s = works(x)
    return (e / s).T

def log_softmax(x):
    x, _, s = works(x)
    return (x - np.log(s)).T