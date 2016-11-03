import numpy as np
import sympy as sm
import theano
import theano.tensor as T

def grad(fun, wrt = None):
    if 'theano' in str(type(fun)):
        return T.grad(fun, [3.0, 1.0])
    elif 'sympy' in str(type(fun)):
        vars = list(fun.free_symbols)
        fns = [sm.lambdify(vars, sm.diff(fun, wrt)) for wrt in vars]

        def eval(x):
            x = np.array(x).astype(float)
            return [fn(*x) for fn in fns]
    return eval

def hess(fun, wrt = None):
    vars = list(fun.free_symbols)
    n = len(vars)
    gs = [sm.diff(fun, wrt) for wrt in vars]
    fns = [sm.lambdify(vars, sm.diff(gs[i], wrt)) for i, wrt in enumerate(vars)]

    def eval(x):
        x = np.array(x).astype(float)
        return np.array([fns[i](*x) for i in xrange(n) for j in xrange(n)]).reshape(n, n)
    return eval