import numpy as np

def generate_w(num_intervals = 10):
    w0 = np.linspace(-100.0, 200.0, num_intervals)
    w1 = np.linspace(-150.0, 150.0, num_intervals)
    return (w0, w1)

def get_best_parameters(grid_w0, grid_w1, grid_losses):
    flat_index = np.argmin(grid_losses)
    index = np.unravel_index(flat_index, grid_losses.shape)
    
    return (grid_losses.flat[flat_index], grid_w0[index[0]], grid_w1[index[1]])
    