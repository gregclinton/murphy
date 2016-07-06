import numpy as np
import scipy.stats as stats
from numpy.linalg import inv

def _rhat(chains):
    results = []
    
    def calc(chains):
        m = len(chains)
        n = len(chains[0])
        psibar = [np.mean(chain) for chain in chains]
        psibardotdot = np.mean(psibar)
        B = n * np.sum((psibar - psibardotdot) ** 2) / (m - 1)
        s2 = [np.sum([(psi - psibar[j]) ** 2 for psi in chain]) / (n - 1) for j, chain in enumerate(chains)]
        W = np.mean(s2)
        V = ((n - 1) * W  + B) / n
        return np.sqrt(V / W)
    
    for j in range(len(chains[0][0])):
        rhat_chains = []    
        for chain in chains:
            chain = np.array(chain)
            l = len(chain) / 2
            rhat_chains.append(chain[0 : l, j])
            rhat_chains.append(chain[l : , j])
        results.append(calc(rhat_chains))
        
    return results

def run(starts, iterations, update):
    def chain(start):
        a = []
        theta = np.copy(start)
        for i in range(iterations):
            update(theta)
            a.append(np.array(theta))
        return a[iterations / 2:]            
            
    chains = [chain(start) for start in starts]
    rhat = np.round(_rhat(chains), 2)
    samples = np.concatenate(chains)
    acceptance = np.round(np.mean(samples[0 : -1] != samples[1 :]), 2)
    return np.sort(samples), rhat, acceptance

def _accept(theta, theta_star, r):
    if stats.uniform.rvs() < min(r, 1):
        for i, t in enumerate(theta_star):
            theta[i] = t
    
def metropolis(log_density, jump):
    def update(theta):
        theta_star = jump(theta)
        r = np.exp(log_density(theta_star) - log_density(theta))
        _accept(theta, theta_star, r)
        
    return update

def hmc(log_density, gradient, M, epsilon = 0.1, L = 10):
    def update(theta):
        M_inv = inv(M)
        phi = stats.multivariate_normal.rvs(np.zeros(len(M)), np.sqrt(M_inv))
        theta_star = np.copy(theta)
        phi_star = np.copy(phi)
        logd = lambda theta, phi: log_density(theta) - np.sum(M_inv.dot(phi ** 2)) / 2
        phi_star += epsilon * gradient(theta) / 2
        for l in range(L):
            theta_star += epsilon * M_inv.dot(phi_star)
            phi_star += epsilon * gradient(theta_star) / (2 if l == L - 1 else 1)
        r = np.exp(logd(theta_star, phi_star) - logd(theta, phi))
        r = 0 if np.isnan(r) else r
        _accept(theta, theta_star, r)
        
    return update