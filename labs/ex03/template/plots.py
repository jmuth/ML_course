# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
#from build_polynomial import *


def plot_fitted_curve(y, x, beta, degree, ax):
    """plot the fitted curve."""
    ax.scatter(x, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(x) - 0.1, max(x) + 0.1, 0.1)
    #tx = np.c_[np.ones((len(xvals), 1)), build_poly(xvals, degree)]
    tx = build_poly(xvals, degree)
    f = tx.dot(beta)
    ax.plot(xvals, f)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial degree " + str(degree))

def build_poly(x, degree):
    """polynomial basis function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    
    #create the matrix tx
    tx = np.ones((x.shape[0], degree+1))
    for i in range(x.shape[0]):
        for j in range(degree+1):
            tx[i, j] = np.power(x[i],j)
    return tx