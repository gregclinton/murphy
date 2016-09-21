import numpy as np

def softmax(w):
    e = np.exp(w.astype(float)).T
    return (e / np.sum(e, axis = 0)).T