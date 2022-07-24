# -*- coding: utf-8 -*-
"""Coil combination
Combine all channels of MRI

"""

import numpy as np
import mripy.signal as sig
from . import util


def sos(img, channel_axis=None):
    """
    Coil combination by SOS (Sum of Squares)

    Parameters
    ----------
    img : ndarray
        multi-channel image, default the last dimension is channel
    channel_axis : None or int

    Returns
    -------
    image_after_combine : ndarray
    """
    ndim = img.ndim
    shape = img.shape
    channel_axis = util.get_channel_axis(channel_axis, ndim)
    output = np.sum(np.abs(img)**2, axis=channel_axis) / shape[channel_axis]
    return np.sqrt(output)