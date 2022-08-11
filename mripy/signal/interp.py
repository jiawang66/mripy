# -*- coding: utf-8 -*-
""" Interpolation functions

Support N-dimensional interpolation.

"""

import numpy as np
from scipy import ndimage

from . import backend, config, util


__all__ = ['zoom', ]


def zoom(data, oshape, dtype=None, order=3, mode='constant',
         cval=0.0, prefilter=True, grid_mode=False):
    """
    Zoom input `data` using spline interpolation.

    Parameters
    ----------
    data : ndarray
        The input array
    oshape : list or tuple of ints
        Shape of output array
    dtype : str, optional
        Dtype of output. By default an array of the same dtype
        as input will be created.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range of 0-5.
    mode : str {‘reflect’, ‘grid-mirror’, ‘constant’, ‘grid-constant’,
        ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’}, optional
        The mode parameter determines how the input array is
        extended beyond its boundaries.
    cval : float, optional
        Value to fill past edges of input if mode is ‘constant’.
        Default is 0.0.
    prefilter : bool, optional
        Determines if the input array is prefiltered with spline_filter
        before interpolation.
    grid_mode : bool, optional
        If False, the distance from the pixel centers is zoomed.

    Returns
    -------
    ndarray

    See Also
    --------
    scipy.ndimage.zoom

    """
    oshape = util.to_tuple(oshape)
    ishape = data.shape

    if len(ishape) != len(oshape):
        raise ValueError('Cannot determine the shape of output.')

    if ishape == oshape:
        return data

    fac = [o / i for i, o in zip(ishape, oshape)]
    output = ndimage.zoom(data, fac, output=dtype, order=order,
                          mode=mode, cval=cval, prefilter=prefilter,
                          grid_mode=grid_mode)
    return output
