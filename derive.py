import numpy as np
from scipy.misc import derivative
import sympy as sm
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
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
def partial(fun, i, x):
    v = x[:]
    def wraps(x):
        v[i] = x
        return fun(v)
    return derivative(wraps, x[i], dx = 1e-6)

def grad(fun):
    if 'sympy' in str(type(fun)):
        vars = list(fun.free_symbols)
        fns = [sm.lambdify(vars, sm.diff(fun, var)) for var in vars]

        def eval(x):
            x = np.array(x).astype(float)
            return [fn(*x) for fn in fns]
    else:
        def eval(x):
            x = np.array(x).astype(float)
            if x.ndim == 0:
                return derivative(fun, x, dx = 1e-6)
            else:
                grad = [partial(fun, i, x) for i in xrange(len(x))]
                return np.array(grad)
    return eval
    
def hess(fun):
    if 'sympy' in str(type(fun)):
        vars = list(fun.free_symbols)
        n = len(vars)
        gs = [sm.diff(fun, var) for var in vars]
        fns = [sm.lambdify(vars, sm.diff(gs[i], var)) for i, var in enumerate(vars)]
        # fns = [sm.lambdify(vars, sm.FunctionMatrix(n, 1, sm.diff(gs[i], var))) for i, var in enumerate(vars)]

        def eval(x):
            x = np.array(x).astype(float)
            hess = np.empty((n, n))

            for i in xrange(n):
                for j in xrange(n): 
                    hess[i, j] = fns[i](*x) # [j]

            return hess   
    elif isinstance(fun, Tensor):
        def eval(vars):
            # http://stackoverflow.com/questions/35266370/tensorflow-compute-hessian-matrix-and-higher-order-derivatives
            cons = lambda x: tf.constant(x, dtype = tf.float32)
            hess = []
            for v1 in vars:
                temp = []
                for v2 in vars:
                    temp.append(tf.gradients(tf.gradients(fun, v2)[0], v1)[0])
                temp = [cons(0) if t == None else t for t in temp] 
                temp = tf.pack(temp)
                hess.append(temp)
            return tf.pack(hess)
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