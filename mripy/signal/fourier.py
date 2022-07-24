# -*- coding: utf-8 -*-
""" Fourier transform

Fast Fourier transform based on FFT

"""

import numpy as np
from . import backend, util


__all__ = ['fftn', 'ifftn']


def fftn(input, oshape=None, axes=None, center=True, norm='ortho'):
    """FFT function that supports centering.

    Args:
        input (ndarray): input array.
        oshape (None or ndarray of ints): output shape.
        axes (None or ndarray of ints): Axes over which to compute the FFT.
        norm (Nonr or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        ndarray: FFT result of dimension oshape.

    See Also:
        :func:`numpy.fft.fftn`
    """
    xp = backend.get_array_module(input)
    if not np.issubdtype(input.dtype, np.complexfloating):
        input = input.astype(np.complex64)

    if center:
        output = _fftnc(input, s=oshape, axes=axes, norm=norm)
    else:
        output = xp.fft.fftn(input, s=oshape, axes=axes, norm=norm)

    if np.issubdtype(input.dtype, np.complexfloating) and input.dtype != output.dtype:
        output = output.astype(input.dtype, copy=False)

    return output


def ifftn(input, oshape=None, axes=None, center=True, norm='ortho'):
    """IFFT function that supports centering.

    Args:
        input (ndarray): input array.
        oshape (None or ndarray of ints): output shape.
        axes (None or ndarray of ints): Axes over which to compute
            the inverse FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        ndarray of dimension oshape.

    See Also:
        :func:`numpy.fft.ifftn`

    """
    xp = backend.get_array_module(input)
    if not np.issubdtype(input.dtype, np.complexfloating):
        input = input.astype(np.complex64)

    if center:
        output = _ifftnc(input, s=oshape, axes=axes, norm=norm)
    else:
        output = xp.fft.ifftn(input, s=oshape, axes=axes, norm=norm)

    if np.issubdtype(input.dtype, np.complexfloating) and input.dtype != output.dtype:
        output = output.astype(input.dtype)

    return output


def _fftnc(a, s=None, axes=None, norm='ortho'):
    """N-D FFT supporting centering

    Compute the N-dimensional discrete Fourier transform with centering

    """
    ndim = a.ndim
    axes = util.normalize_axes(axes, ndim)
    xp = backend.get_array_module(a)

    if s is None:
        s = a.shape

    res = util.resize(a, s)
    res = xp.fft.fftshift(res, axes=axes)
    res = xp.fft.fftn(res, axes=axes, norm=norm)
    res = xp.fft.ifftshift(res, axes=axes)
    return res


def _ifftnc(a, s=None, axes=None, norm='ortho'):
    """N-D IFFT supporting centering

    Compute the N-dimensional inverse discrete Fourier transform with centering

    """
    ndim = a.ndim
    axes = util.normalize_axes(axes, ndim)
    xp = backend.get_array_module(a)

    if s is None:
        s = a.shape

    res = util.resize(a, s)
    res = xp.fft.fftshift(res, axes=axes)
    res = xp.fft.ifftn(res, axes=axes, norm=norm)
    res = xp.fft.ifftshift(res, axes=axes)
    return res