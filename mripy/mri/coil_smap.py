# -*- coding: utf-8 -*-
"""Coil Sensitivity Map
Compute soil sensituvity map

"""

from . import coil_com, util
from ..signal import backend


def smap(img, img_com=None, caxis=None):
    """
    Compute the sensitivity map based on SOS

    Parameters
    ----------
    img : ndarray
        multi-channel image
    img_com : None or ndarray
        image after coil combination
    caxis : int
        channel axis, default the last dimension

    Returns
    -------
    sensitivity_map : ndarray
    """
    ndim = img.ndim
    xp = backend.get_array_module(img)
    caxis = util.get_channel_axis(caxis, ndim)

    if img_com is None:
        img_com = coil_com.sos(img, caxis=caxis)

    if not img.ndim == img_com.ndim:
        img_com = xp.expand_dims(img_com, caxis)

    indices = img_com == 0
    img_com[indices] = 1
    return img / img_com
