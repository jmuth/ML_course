# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def compute_e(y, tx, w):
    ''' Computation of e according to the lesson week 2
    '''
    return (y - np.dot(tx,w))

def compute_N(e):
    ''' Size of e for normalization (theory lesson week 2)
    '''
    return e.shape[0]

def compute_cost(y, tx, w):
    """calculate the cost.

    you can calculate the cost by mse or mae.
    """
    e = compute_e(y, tx, w)                      # vector e
    N = compute_N(e)                             # size
    L_MSE = np.dot(np.matrix.transpose(e), e)
    L_MSE = L_MSE / (2 * N)                      # normalization
    
    return L_MSE

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
