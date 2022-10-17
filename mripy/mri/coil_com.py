# -*- coding: utf-8 -*-
"""Coil combination
Combine all channels of MRI

"""

from . import util
from ..signal import backend


def sos(img, caxis=None):
    """
    Coil combination by SOS (Sum of Squares)

    Parameters
    ----------
    img : ndarray
        multi-channel image, default the last dimension is channel
    caxis : None or int

    Returns
    -------
    image_after_combine : ndarray
    """
    ndim = img.ndim
    shape = img.shape
    xp = backend.get_array_module(img)

    caxis = util.get_channel_axis(caxis, ndim)
    output = xp.sum(xp.abs(img)**2, axis=caxis) / shape[caxis]
    return xp.sqrt(output)