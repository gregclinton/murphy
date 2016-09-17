import matplotlib.pyplot as plt
import numpy as np

colors = ['r', 'b', 'g']

def classes(y):
    return range(len(np.unique(y)))
    
def limits(X):
    plt.plot(X[:, 0], X[:, 1], alpha = 0)
    axis = plt.gca()
    minx, maxx = axis.get_xlim()
    miny, maxy = axis.get_ylim()
    return minx, maxx, miny, maxy

def mesh(X):
    minx, maxx, miny, maxy = limits(X)
    gridx = np.linspace(minx, maxx, 51)
    gridy = np.linspace(miny, maxy, 51)
    meshx, meshy = np.meshgrid(gridx, gridy)
    return meshx, meshy, np.vstack([meshx.flatten(), meshy.flatten()]).T

def show_points(X, y):
    for c in classes(y):
        i = y == c
        color = colors[c]
        plt.plot(X[i, 0], X[i, 1], 'o', markeredgewidth = 1.5, markerfacecolor = 'none', markeredgecolor = color)

def show_regions(clf, X):
    meshx, meshy, XX = mesh(X)
    Z = clf.predict(XX)
    plt.contour(meshx, meshy, Z.reshape(meshx.shape), levels = classes(Z))
    ZZ = clf.predict_log_proba(XX)
    plt.contour(meshx, meshy, (ZZ[:, 1] - ZZ[:, 0]).reshape(meshx.shape))

    for c in classes(Z):
        i = Z == c
        color = colors[c]
        plt.plot(XX[i, 0], XX[i, 1], 'o', alpha = 0.2, markersize = 2.5, markerfacecolor = color, markeredgecolor = color)