from scipy import stats

def frozen(df, scale):
    return stats.invgamma(df / 2.0, scale = (scale ** 2 * df if scale else 1) / 2.0)

def pdf(x, df, scale = None):
    return frozen(df, scale).pdf(x)

def logpdf(x, df, scale = None):
    return frozen(df, scale).logpdf(x)

def rvs(df, scale = None, size = 1):
    return frozen(df, scale).rvs(size = size)

def rvs(df, scale = None, size = 1):
    return df * scale ** 2 / stats.chi2.rvs(df, size = size)