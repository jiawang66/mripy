# -*- coding: utf-8 -*-
""" Convolution functions with multi-dimension, and multi-channel support.

"""

import numpy as np
import scipy.signal as signal
from . import config, backend, util


def convolve(data, filt, mode='full', strides=None, multi_channel=False):
    r"""
    Convolution that supports multi-dimensional and multi-channel inputs.
    Based on scipy.signal.convolve and cudnn.

    Parameters
    ----------
    data : ndarray
        data array of shape:
        :math:`[..., m_1, ..., m_D]` if multi_channel is False,
        :math:`[..., c_i, m_1, ..., m_D]` otherwise.
    filt : ndarray
        filter array of shape:
        :math:`[n_1, ..., n_D]` if multi_channel is False
        :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
    mode : str {'full', 'same', 'valid'}, optional

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    strides : None or list or tuple of ints
        convolution strides of length D.
    multi_channel : bool
        specify if input/output has multiple channels.

    Returns
    -------
    output : ndarray
        output array of shape:
        :math:`[..., p_1, ..., p_D]` if multi_channel is False,
        :math:`[..., c_o, p_1, ..., p_D]` otherwise.

    See Also
    --------
    scipy.signal.convolve
    """
    xp = backend.get_array_module(data)
    if xp == np:
        output = _convolve(data, filt, mode=mode, strides=strides,
                           multi_channel=multi_channel)
    else:
        raise NotImplementedError('Currently not support convolution for cupy.')

    return output


def convolve_data_adjoint(output, filt, data_shape,
                          mode='full', strides=None, multi_channel=False):
    r"""
    Adjoint convolution operation with respect to data.

    Parameters
    ----------
    output : ndarray
        An array of the resolution of conv(data, filt).
        output array of shape
        :math:`[..., p_1, ..., p_D]` if multi_channel is False,
        :math:`[..., c_o, p_1, ..., p_D]` otherwise.
    filt : ndarray
        filt array of shape
        :math:`[n_1, ..., n_D]` if multi_channel is False
        :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
    data_shape : list or tuple
        the shape of data of length D
    mode : str {'full', 'same', 'valid'}, optional
    strides : None or list or tuple of ints
        convolution strides of length D.
    multi_channel : bool
        specify if input/output has multiple channels.

    Returns
    -------
    data : ndarray
        data array of shape:
        :math:`[..., m_1, ..., m_D]` if multi_channel is False,
        :math:`[..., c_i, m_1, ..., m_D]` otherwise.
    """
    data_shape = tuple(data_shape)

    xp = backend.get_array_module(output)
    if xp == np:
        data = _convolve_data_adjoint(output, filt, data_shape,
                                      mode=mode, strides=strides,
                                      multi_channel=multi_channel)
    else:
        raise NotImplementedError('Currently not support convolution adjoint for cupy.')

    return data


def convolve_filter_adjoint(output, data, filt_shape,
                            mode='full', strides=None, multi_channel=False):
    r"""
    Adjoint convolution operation with respect to filter.

    Parameters
    ----------
    output : ndarray
        An array of the resolution of conv(data, filt).
        output array of shape
        :math:`[..., p_1, ..., p_D]` if multi_channel is False,
        :math:`[..., c_o, p_1, ..., p_D]` otherwise.
    data : ndarray
        data array of shape:
        :math:`[..., m_1, ..., m_D]` if multi_channel is False,
        :math:`[..., c_i, m_1, ..., m_D]` otherwise.
    filt_shape : list or tuple
        the shape of filter of length D
    mode : str {'full', 'same', 'valid'}, optional
    strides : None or list or tuple of ints
        convolution strides of length D.
    multi_channel : bool
        specify if input/output has multiple channels.

    Returns:
    filt : ndarray
        filtlter array of shape
        :math:`[n_1, ..., n_D]` if multi_channel is False
        :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
    """
    filt_shape = tuple(filt_shape)
    xp = backend.get_array_module(data)
    if xp == np:
        filt = _convolve_filter_adjoint(output, data, filt_shape,
                                        mode=mode, strides=strides,
                                        multi_channel=multi_channel)
    else:  # pragma: no cover
        raise NotImplementedError('Currently not support convolution adjoint for cupy.')

    return filt


def _convolve(data, filt, mode='full', strides=None, multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(data.shape, filt.shape,
                                                         mode, strides, multi_channel)

    # Normalize shapes.
    data = data.reshape((B, c_i) + m)
    filt = filt.reshape((c_o, c_i) + n)
    output = np.zeros((B, c_o) + p, dtype=data.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    # Why multi-channel need c_i and c_o?
    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                # why self add?
                output[k, j] += signal.convolve(
                    data[k, i], filt[j, i], mode=mode)[slc]

    # Reshape.
    if multi_channel:
        output = output.reshape(b + (c_o, ) + p)
    else:
        output = output.reshape(b + p)

    return output


def _convolve_data_adjoint(output, filt, data_shape,
                           mode='full', strides=None,
                           multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(data_shape, filt.shape,
                                                         mode, strides, multi_channel)

    # Normalize shapes.
    output = output.reshape((B, c_o) + p)
    filt = filt.reshape((c_o, c_i) + n)
    data = np.zeros((B, c_i) + m, dtype=output.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    adjoint_mode = None
    output_kj = None

    if mode == 'full':
        output_kj = np.zeros([m_d + n_d - 1 for m_d, n_d in zip(m, n)], dtype=output.dtype)
        adjoint_mode = 'valid'
    elif mode == 'valid':
        output_kj = np.zeros([max(m_d, n_d) - min(m_d, n_d) + 1 for m_d, n_d in zip(m, n)],
                             dtype=output.dtype)
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = 'full'
        else:
            adjoint_mode = 'valid'
    elif mode == 'same':
        raise NotImplementedError(f'Not support for mode {mode}')
        # output_kj = np.zeros([m_d for m_d, n_d in zip(m, n)], dtype=output.dtype)
        # adjoint_mode = 'same'

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                data[k, i] += signal.correlate(output_kj, filt[j, i], mode=adjoint_mode)

    # Reshape.
    data = data.reshape(data_shape)
    return data


def _convolve_filter_adjoint(output, data, filt_shape,
                             mode='full', strides=None,
                             multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(data.shape, filt_shape,
                                                         mode, strides, multi_channel)

    # Normalize shapes.
    data = data.reshape((B, c_i) + m)
    output = output.reshape((B, c_o) + p)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    adjoint_mode = None
    output_kj = None

    if mode == 'full':
        output_kj = np.zeros([m_d + n_d - 1 for m_d, n_d in zip(m, n)],
                             dtype=output.dtype)
        adjoint_mode = 'valid'
    elif mode == 'valid':
        output_kj = np.zeros([max(m_d, n_d) - min(m_d, n_d) + 1 for m_d, n_d in zip(m, n)],
                             dtype=output.dtype)
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = 'valid'
        else:
            adjoint_mode = 'full'
    elif mode == 'same':
        raise NotImplementedError(f'Not support for mode {mode}')
        # output_kj = np.zeros([m_d for m_d, n_d in zip(m, n)], dtype=output.dtype)
        # adjoint_mode = 'same'

    filt = np.zeros((c_o, c_i) + n, dtype=output.dtype)
    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                filt[j, i] += signal.correlate(output_kj, data[k, i], mode=adjoint_mode)

    # Reshape.
    filt = filt.reshape(filt_shape)
    return filt


def _get_convolve_params(data_shape, filt_shape,
                         mode, strides, multi_channel):
    D = len(filt_shape) - 2 * multi_channel     # ndim of data or filt without multi channel
    m = tuple(data_shape[-D:])                  # data shape without multi channel
    n = tuple(filt_shape[-D:])                  # filt shape without multi channel
    b = tuple(data_shape[:-D - multi_channel])  # what is the role of b? b is a always empty tuple ().
    B = util.prod(b)                            # B = 1 ?

    if multi_channel:
        if filt_shape[-D - 1] != data_shape[-D - 1]:
            raise ValueError(f'Data channel mismatch, '
                             f'got {data_shape[-D - 1]} from data '
                             f'and {filt_shape[-D - 1]} from filt.')

        c_i = filt_shape[-D - 1]
        c_o = filt_shape[-D - 2]
    else:
        c_i = 1
        c_o = 1

    if strides is None:
        s = (1, ) * D
    else:
        if len(strides) != D:
            raise ValueError(f'Strides must have length {D}.')

        s = tuple(strides)

    # determine the shape of output
    if mode == 'full':
        p = tuple((m_d + n_d - 1 + s_d - 1) // s_d
                  for m_d, n_d, s_d in zip(m, n, s))
    elif mode == 'valid':
        if (any(m_d >= n_d for m_d, n_d in zip(m, n)) and
            any(m_d < n_d for m_d, n_d in zip(m, n))):
            raise ValueError('In valid mode, either data or filter must be '
                             'at least as large as the other in every axis.')

        p = tuple((m_d - n_d + 1 + s_d - 1) // s_d
                  for m_d, n_d, s_d in zip(m, n, s))
    elif mode =='same':
        if (any(m_d >= n_d for m_d, n_d in zip(m, n)) and
            any(m_d < n_d for m_d, n_d in zip(m, n))):
            raise ValueError('In valid mode, either data or filter must be '
                             'at least as large as the other in every axis.')
        p = tuple((m_d + s_d - 1) // s_d
                  for m_d, n_d, s_d in zip(m, n, s))
    else:
        raise ValueError('Invalid mode, got {}'.format(mode))

    return D, b, B, m, n, s, c_i, c_o, p
