# -*- coding: utf-8 -*-
"""Utility functions

"""

import mripy.signal as sig


def get_channel_axis(channel_axis, ndim):
    if channel_axis is None:
        channel_axis = ndim - 1

    return sig.util.normalize_axes((channel_axis,),ndim)[0]
