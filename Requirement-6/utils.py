import numpy as np

# Implemented for you as a gift
def batchify(x_data, y_data, batch_size):
    '''
    Given x_data of shape (m, n) and y_data of shape (m) return x_data of shape (batch_size, m/batch_size, n) and y_data of shape (batch_size, m/batch_size)
    '''
    m = x_data.shape[0]
    
    # shuffle the data with numpy's permutation function.
    index = np.random.permutation(m)
    x_data, y_data = x_data[index], y_data[index]
    
    # make y_data a one-hot vector
    u = len(np.unique(y_data))
    
    # truncate the data to be divisible by the batch size
    x_data = x_data[:m - m%batch_size]
    y_data = y_data[:m - m%batch_size]
    
    # batchify the data using numpy
    x_data = np.split(x_data, m//batch_size)
    y_data = np.split(y_data, m//batch_size)
    
    return np.array(x_data), np.array(y_data)