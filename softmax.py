import numpy as np

def softmax(x):
    x = np.array(x, dtype = float)
    x -= np.max(x) # prevent overflow
    e = np.exp(x).T
    return (e / np.sum(e, axis = 0)).T