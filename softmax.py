import numpy as np

def softmax(a):
    e = np.exp(np.array(a, dtype = float)).T
    return (e / np.sum(e, axis = 0)).T