# -*- coding: utf-8 -*-
"""Coil combination
Combine all channels of MRI

"""

from . import util
from ..signal import backend


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
    xp = backend.get_array_module(img)

    channel_axis = util.get_channel_axis(channel_axis, ndim)
    output = xp.sum(xp.abs(img)**2, axis=channel_axis) / shape[channel_axis]
    return xp.sqrt(output)