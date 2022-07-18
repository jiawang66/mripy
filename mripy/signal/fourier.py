# -*- coding: utf-8 -*-
""" Fourier transform

Fast Fourier transform based on FFT

"""

import numpy as np
from math import ceil
from . import backend, util


def _fftnc(a, oshape=None, axes=None, norm=None):
    """N-D FFT supporting centering

    Compute the N-dimensional discrete Fourier transform with centering

    """
    ndim = a.ndim
    axes = util.normalize_axes(axes, ndim)
    xp = backend.get_array_module(a)

    if oshape is None:
        oshape = a.shape

    res = util.resize(a, oshape)
    res = xp.fft.fftshift(res, axes=axes)
    res = xp.fft.fftn(res, axes=axes, norm=norm)
    res = xp.fft.ifftshift(res, axes=axes)
    return res
