import numpy as np
from scipy import stats

def get(path, params):
    if path == 'charts':
        mvn = stats.multivariate_normal([80, 80], 240 * np.eye(2)).pdf
        x = y = list(np.linspace(60, 120, 101));
        z = [[mvn([xx, yy]) for yy in y] for xx in x]
        
        return {
            'line': { 'x': [60, 80, 120], 'y': [60, 60, 120] },
            'contour': { 'x': x, 'y': y, 'z': z }
        }
    elif path == 'next':
        return [100, 100, 200]
    return []