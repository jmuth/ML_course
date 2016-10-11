# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    
    # Compute optimal weights
    xx = np.dot(np.transpose(tx),tx)

    bxx = xx + lamb*np.identity(len(xx))

    try:
        inv = np.linalg.inv(bxx)
    except:
        raise ValueError("Matrix X^TX not invertible") 

    xy = np.dot(np.transpose(tx),y)
    w_star = np.dot(inv, xy)
   
    return w_star