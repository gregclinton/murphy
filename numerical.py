import numpy as np

# http://localhost:8888/edit/Desktop/assignment1/cs231n/gradient_check.py

def grad(f, x):
    x = np.array(x).astype(float)
    h = 0.00001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval
        grad[ix] = (fxph - fxmh) / (2 * h)
        it.iternext()
        
    return grad

def hess(f, x):
    return 0