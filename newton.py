import scipy.optimize as opt
from converge import Converge
from numpy.linalg import inv

def minimize(f, theta, g, H, epsilon = 0.0000001, maxsteps = 100):
    converge = Converge(f, epsilon, maxsteps)
    
    while not converge.done(theta):
        d = -inv(H(theta)).dot(g(theta))
        eta = opt.minimize(lambda eta: f(theta + eta * d), 1).x
        theta += eta * d
    return theta