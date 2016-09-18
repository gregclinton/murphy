import pandas as pd
import numpy as np

def get():
    url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
    data = pd.read_csv(url)
    species = list(np.unique(data.species))
    y = np.array([species.index(s) for s in data.species])
    X = data.values[:, 0:-1]
    return X * 1.0, y * 1.0    