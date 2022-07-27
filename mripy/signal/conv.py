# -*- coding: utf-8 -*-
""" Convolution functions with multi-dimension, and multi-channel support.

"""

import numpy as np
import scipy.signal as signal
from . import config, backend, util


def convolve(data, filt, mode='full'):
    """
    Convolution

    Parameters
    ----------
    data
    filt
    mode

    Returns
    -------

    """
    pass


def convolve_data_adjoint(output, filt, data_shape, mode='full'):
    """
    Adjoint of convolution with respect to data

    Parameters
    ----------
    output : ndarray
    filt : ndarray
    data_shape : int or list or tuple
    mode: str {'full', 'valid', 'same'}, optional

    Returns
    -------
    ndarray
    """
    filt_shape = tuple(k if k % 2 == 1 else k + 1 for k in filt.shape)
    filt = util.resize(filt, filt_shape)
    filt = _reverse_and_conj(filt)

    if mode == 'full' or mode == 'valid':
        adjoint_mode = 'valid'

        # zero pad the output
        pad_shape = [i + j - 1 for i, j in zip(data_shape, filt_shape)]
        num_left = [int(np.ceil((i - j) / 2)) for i, j in zip(pad_shape, output.shape)]
        num_right = [(i - j) // 2 for i, j in zip(pad_shape, output.shape)]
        dims = list(range(output.ndim))
        output = util.crop_pad(output, dims=dims, num_left=num_left, num_right=num_right)

    elif mode == 'same':
        adjoint_mode = 'same'
    else:
        raise ValueError(f'Unknown mode, got {mode}')

    data = signal.convolve(output, filt, mode=adjoint_mode)
    return data


def convolve_filt_adjoint(output, data, filt_shape, mode='full'):
    """
    Adjoint of convolution with respect to filter

    Parameters
    ----------
    output : ndarray
    data : ndarray
    filt_shape : int or list or tuple
    mode: str {'full', 'valid', 'same'}, optional

    Returns
    -------
    ndarray
    """
    if mode == 'full' or mode == 'valid':
        return convolve_data_adjoint(output, data, filt_shape, mode=mode)
    elif mode == 'same':
        adjoint_mode = 'valid'
        flag = [True if i % 2 == 1 and j % 2 == 0 else False
                for i, j in zip(data.shape, filt_shape)]
        output_shape = [i + j - 1if f else i
                        for i, j, f in zip(data.shape, filt_shape, flag)]
        output = util.resize(output, output_shape)

        return convolve_data_adjoint(output, data, filt_shape, mode=adjoint_mode)
    else:
        raise ValueError(f'Unknown mode, got {mode}')


def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()