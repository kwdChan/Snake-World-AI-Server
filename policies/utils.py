import numpy as np
from typing import Tuple
def to_sparse(arr, x_bounds:Tuple, y_bounds: Tuple):
    """
    if x_bounds = (0, 2)
    size = 2
    included indx: 0, 1
    """

    x_size = x_bounds[1]-x_bounds[0]
    y_size = y_bounds[1]-y_bounds[0]

    result = np.zeros((x_size, y_size))
    if not len(arr):
        return result
    arr[..., 0] -= x_bounds[0]
    arr[..., 1] -= y_bounds[0]

    #return arr
    inbound_filter = np.apply_along_axis(lambda a: ((a[0]>=0) & (a[0]<x_size) & (a[1]>=0) & (a[1] <y_size)), axis=1, arr=arr)

    def assign(a):
         result[a[0], a[1]] = 1
    
    arr = arr[inbound_filter,...]
    
    if not len(arr):
        return result
    #return arr
    np.apply_along_axis(assign, axis =1, arr=arr)

    return result
