# -*- coding: utf-8 -*-
"""Sample
K-space sampling function

"""

import math
import numpy as np
from ..signal import util as sig_util


def uniform_cartesian(img_shape, rate=1, axis=0):
    """
    Uniform Cartesian downsampling in k-space along `dim` with `rate` of acceleration.

    Parameters
    ----------
    img_shape : tuple or list of ints
        length-N of N-D image shape
    rate : int
        Target acceleration factor. Must be laeger than 1. Default 1.
    axis : int
        Acceleration dimension. Default 0.

    Returns
    -------
    mask : ndarray
        uniform Cartesian k-space sampling mask

    """
    if rate < 1 or rate % 1 != 0:
        raise ValueError(f'rate must be a integer equal or larger than 1, got {rate}')

    img_shape = sig_util.to_tuple(img_shape)
    ndim = len(img_shape)
    axis = axis % ndim

    oslice = tuple([slice(0, s, rate) if i == axis else slice(0, s) for i, s in enumerate(img_shape)])
    mask = np.zeros(img_shape, dtype=int)
    mask[oslice] = 1
    return mask
