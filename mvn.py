from scipy import stats
import numpy as np

def rvs(mu, Sigma):
    z = stats.norm.rvs(size = len(mu))
    return mu + np.linalg.cholesky(Sigma).dot(z)

def rvs(mu, Sigma):
    # from stober https://gist.github.com/stober/4964727
    # better numerical stability than choleskey for near-singular Sigma
    z = stats.norm.rvs(size = len(mu))
    [U, S, V] = np.linalg.svd(Sigma)
    A = U * np.sqrt(S)
    return mu + A.dot(z)

# mvn = stats.multivariate_normal