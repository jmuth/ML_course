# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_cost(y, tx, beta):
    """compute the loss by mse."""
    e = y - tx.dot(beta)
    mse = e.dot(e) / (2 * len(e))
    return mse


def least_squares(y, tx):
    """calculate the least squares solution."""
    xx = np.dot(np.matrix.transpose(tx), tx)
    
    # handle non-inversable matrix case
    try:
        inv_xx = np.linalg.inv(xx)
    except: 
        raise ValueError('Matrix xx is not invertible')
    
    w = np.matrix.dot(np.matrix.dot(inv_xx, np.matrix.transpose(tx)), y)
    loss = compute_cost(y, tx, w)
    
    return loss, w
