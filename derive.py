import numpy as np
from scipy.misc import derivative
import sympy as sm
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
import theano
import theano.tensor as T
# import numdifftools as nd

def partial(fun, vars = None):
    def eval(x, i):
        # http://stackoverflow.com/questions/20708038/scipy-misc-derivative-for-mutiple-argument-function
        v = x[:]
        def wraps(x):
            v[i] = x
            return fun(v)    
        return derivative(wraps, x[i], dx = 1e-6)
    return eval

def grad(fun, vars = None):
    if isinstance(fun, Tensor):
        return tf.pack([tf.gradients(fun, var)[0] for var in vars])
    elif 'theano' in str(type(fun)):
        return None
    elif 'sympy' in str(type(fun)):
        vars = list(fun.free_symbols)
        fns = [sm.lambdify(vars, sm.diff(fun, var)) for var in vars]

        def eval(x):
            x = np.array(x).astype(float)
            return [fn(*x) for fn in fns]
    else:
        part = partial(fun, vars)
        
        def eval(x):
            x = np.array(x).astype(float)
            return np.array([part(x, i) for i in xrange(len(x))])
    return eval

def hess(fun, vars = None):
    if isinstance(fun, Tensor):
        # http://stackoverflow.com/questions/35266370/tensorflow-compute-hessian-matrix-and-higher-order-derivatives
        def row(v1):
            row = [tf.gradients(tf.gradients(fun, v2)[0], v1)[0] for v2 in vars]
            row = [tf.constant(0.0) if t == None else t for t in row] 
            return tf.pack(row)
        return tf.pack([row(v1) for v1 in vars])
    elif 'theano' in str(type(fun)):
        return None
    elif 'sympy' in str(type(fun)):
        vars = list(fun.free_symbols)
        n = len(vars)
        gs = [sm.diff(fun, var) for var in vars]
        fns = [sm.lambdify(vars, sm.diff(gs[i], var)) for i, var in enumerate(vars)]

        def eval(x):
            x = np.array(x).astype(float)
            return np.array([fns[i](*x) for i in xrange(n) for j in xrange(n)]).reshape(n, n)
    else:
        part = partial(grad(fun, vars), vars)

        def eval(x):
            x = np.array(x).astype(float)
            n = len(x)
            return np.array([part(x, i)[j] for i in xrange(n) for j in xrange(n)]).reshape((n, n))    
    return eval

def my_numeric_grad(fun):
    def eval(x):
        # http://localhost:8888/edit/Desktop/assignment1/cs231n/gradient_check.py
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