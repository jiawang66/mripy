# -*- coding: utf-8 -*-
"""Coil Sensitivity Map
Compute soil sensituvity map

"""

import mripy.signal as sig
from . import coil_com, util


def smap(img, img_com=None, channel_axis=None):
    """
    Compute the sensitivity map based on SOS

    Parameters
    ----------
    img : ndarray
        multi-channel image
    img_com : None or ndarray
        image after coil combination
    channel_axis : int
        channel axis, default the last dimension

    Returns
    -------
    sensitivity_map : ndarray
    """
    ndim = img.ndim
    xp = sig.backend.get_array_module(img)
    channel_axis = util.get_channel_axis(channel_axis, ndim)

    if img_com is None:
        img_com = coil_com.sos(img, channel_axis=channel_axis)

    if not img.ndim == img_com.ndim:
        img_com = xp.expand_dims(img_com, channel_axis)

    indices = img_com == 0
    img_com[indices] = 1
    return img / img_com
