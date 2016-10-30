import numpy as np
import numdifftools as nd
from scipy.misc import derivative
import tensorflow as tf      
import theano
import theano.tensor as T

# import numdifftools as nd

# http://localhost:8888/edit/Desktop/assignment1/cs231n/gradient_check.py

def grad(fun):
    def eval(x):
        x = np.array(x).astype(float)
        h = 1e-6
        grad = np.zeros_like(x)
        it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])

        while not it.finished:
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h
            fxph = fun(x)
            x[ix] = oldval - h
            fxmh = fun(x)
            x[ix] = oldval
            grad[ix] = (fxph - fxmh) / (2 * h)
            it.iternext()

        return grad
    return eval

# http://stackoverflow.com/questions/20708038/scipy-misc-derivative-for-mutiple-argument-function
def partial(func, ix, point):
    v = point[:]
    def wraps(x):
        v[ix] = x
        return func(v)
    return derivative(wraps, point[ix], dx = 1e-6)

def grad(fun):
    def eval(x):
        x = np.array(x)
        if x.ndim == 0:
            return derivative(fun, x, dx = 1e-6)
        else:
            grad = [partial(fun, i, x) for i in xrange(len(x))]
            return np.array(grad)
    return eval
    
def hess(fun):
    if True:
        def eval(vars):
            # http://stackoverflow.com/questions/35266370/tensorflow-compute-hessian-matrix-and-higher-order-derivatives
            cons = lambda x: tf.constant(x, dtype = tf.float32)
            mat = []
            for v1 in vars:
                temp = []
                for v2 in vars:
                    temp.append(tf.gradients(tf.gradients(fun, v2)[0], v1)[0])
                temp = [cons(0) if t == None else t for t in temp] 
                temp = tf.pack(temp)
                mat.append(temp)
            return tf.pack(mat)
    else:
        g = grad(fun)

        def eval(x):
            x = np.array(x).astype(float)
            n = len(x)
            hess = np.empty((n, n))

            for i in xrange(n):
                for j in xrange(n): 
                    hess[i, j] = partial(g, i, x)[j]

            return hess   
    return eval