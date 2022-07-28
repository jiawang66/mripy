# -*- coding: utf-8 -*-
""" Convolution functions

Support N-dimension convolution with strides not equal to 1

"""

import warnings
import numpy as np
import scipy.signal as signal
from . import backend, util


__all__ = ['convolve', 'convolve_data_adjoint', 'convolve_filt_adjoint']


def convolve(data, filt, mode='full', strides=None):
    """
    Convolve two N-dimensional arrays.

    Convolve `data` and `filt`, with the output size determined by
    the `mode` argument.

    Parameters
    ----------
    data : ndarray
        First input
    filt : ndarray
        Second input. Should be have the same number of dimensions as `data`.
    mode : str {`full`, `valid`, `same`}, optional
        A string indicating the size of the output

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
    strides : list or tuple of ints, None
        convolution strides of length of the number of dimension

    Returns
    -------
    output : ndarray
        An N-dimensional array containing a subset of the discrete linear
        convolution of `data` with `filt`.

    See Also
    --------
    convolve_data_adjoint
    convolve_filt_adjoint

    """
    xp = backend.get_array_module(data)

    if xp == np:
        output = _convolve(data, filt, mode=mode, strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(data)
        data = backend.to_device(data)
        filt = backend.to_device(filt)
        output = _convolve(data, filt, mode=mode, strides=strides)
        output = backend.to_device(output, device)

    return output


def convolve_data_adjoint(output, filt, data_shape, mode='full', strides=None):
    """
    Adjoint of convolution with respect to `data`

    `output` = convolution of `data` with `filt`

    Parameters
    ----------
    output : ndarray
        Fist input. An N-dimensional array containing a subset of the
        discrete linear convolution of `data` with `filt`.
    filt : ndarray
        Second input. Should be have the same number of dimensions as `output`.
    data_shape : list or tuple of ints
        shape of `data`
    mode : str {`full`, `valid`, `same`}, optional
    strides : list or tuple of ints, None
        convolution strides of length of the number of dimension

    Returns
    -------
    data : ndarray
        An N-dimensional array with shape of `data_shape`

    See Also
    --------
    convolve
    convolve_filt_adjoint

    """
    data_shape = tuple(data_shape)
    xp = backend.get_array_module(output)

    if xp == np:
        data = _convolve_data_adjoint(output, filt, data_shape,
                                      mode=mode, strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(output)
        output = backend.to_device(output)
        filt = backend.to_device(filt)
        data = _convolve_data_adjoint(output, filt, data_shape,
                                      mode=mode, strides=strides)
        data = backend.to_device(data, device)

    return data


def convolve_filt_adjoint(output, data, filt_shape, mode='full', strides=None):
    """
    Adjoint of convolution with respect to `filt`

    `output` = convolution of `data` with `filt`

    Parameters
    ----------
    output : ndarray
        Fist input. An N-dimensional array containing a subset of the
        discrete linear convolution of `data` with `filt`.
    data : ndarray
        Second input. Should be have the same number of dimensions as `output`.
    filt_shape : list or tuple of ints
        shape of `filt`
    mode : str {`full`, `valid`, `same`}, optional
    strides : list or tuple of ints, None
        convolution strides of length of the number of dimension

    Returns
    -------
    filt : ndarray
        An N-dimensional array with shape of `filt_shape`

    See Also
    --------
    convolve
    convolve_data_adjoint

    """
    filt_shape = tuple(filt_shape)
    xp = backend.get_array_module(output)

    if xp == np:
        filt = _convolve_filt_adjoint(output, data, filt_shape,
                                      mode=mode, strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(output)
        output = backend.to_device(output)
        data = backend.to_device(data)
        filt = _convolve_filt_adjoint(output, data, filt_shape,
                                      mode=mode, strides=strides)
        filt = backend.to_device(filt, device)

    return filt


def _convolve(data, filt, mode='full', strides=None):
    s, p = _get_convolve_params(data.shape, filt.shape, mode, strides)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    output = signal.convolve(data, filt, mode=mode)[slc]
    return output


def _convolve_data_adjoint(output, filt, data_shape,
                           mode='full', strides=None):
    s, p = _get_convolve_params(data_shape, filt.shape, mode, strides)

    # if stride > 1, zero-fill the output
    if not s == (1,) * len(s):
        slc = tuple(slice(None, None, s_d) for s_d in s)
        output_fill = np.zeros(p, dtype=output.dtype)
        output_fill[slc] = output
        output = output_fill

    # zero-padding of the even-size dimensions to be odd-size
    filt_shape = tuple(k if k % 2 == 1 else k + 1 for k in filt.shape)
    filt = util.resize(filt, filt_shape)

    # reverse the filter
    filt = _reverse_and_conj(filt)

    if mode == 'full' or mode == 'valid':
        adjoint_mode = 'valid'

        # zero pad the output to have `full` size
        pad_shape = tuple([i + j - 1 for i, j in
                           zip(data_shape, filt_shape)])
        if not output.shape == pad_shape:
            # Pad on the left first.
            # That is, one more zero on the left than on the right
            num_left = [int(np.ceil((i - j) / 2)) for i, j in
                        zip(pad_shape, output.shape)]
            num_right = [(i - j) // 2 for i, j in
                         zip(pad_shape, output.shape)]
            dims = list(range(output.ndim))
            output = util.crop_pad(output, dims=dims,
                                   num_left=num_left,
                                   num_right=num_right)
    else:
        adjoint_mode = 'same'

    data = signal.convolve(output, filt, mode=adjoint_mode)
    return data


def _convolve_filt_adjoint(output, data, filt_shape,
                           mode='full', strides=None):
    s, p = _get_convolve_params(data.shape, filt_shape, mode, strides)

    # if stride > 1, zero-fill the output
    # and then set the strides = None
    if not s == (1,) * len(s):
        slc = tuple(slice(None, None, s_d) for s_d in s)
        output_fill = np.zeros(p, dtype=output.dtype)
        output_fill[slc] = output
        output = output_fill
        strides = None

    if mode == 'full' or mode == 'valid':
        return _convolve_data_adjoint(output, data, filt_shape,
                                      mode=mode, strides=strides)
    elif mode == 'same':
        adjoint_mode = 'full'
        flag = [True if i % 2 == 1 and j % 2 == 0 else False
                for i, j in zip(data.shape, filt_shape)]
        output_shape = [i + j - 1if f else i
                        for i, j, f in zip(data.shape, filt_shape, flag)]
        output = util.resize(output, output_shape)

        return _convolve_data_adjoint(output, data, filt_shape,
                                      mode=adjoint_mode, strides=strides)
    else:
        raise ValueError(f'Invalid mode, got {mode}')


def _get_convolve_params(data_shape, filt_shape, mode, strides):
    """
    Check the convolution params and return the strides.
    """
    m = tuple(data_shape)
    n = tuple(filt_shape)
    D = len(m)

    if not len(m) == len(n):
        raise ValueError(f'The number of dimensions of `data` and `filt` not match. '
                         f'the dimension of `data` is {len(m)}, '
                         f'the dimension of `filt` is {len(n)}.')

    if strides is None:
        s = (1,) * D
    else:
        if len(strides) != D:
            raise ValueError(f'Strides must have length {D}.')

        s = tuple(strides)

    if mode == 'full':
        p = tuple([i + j - 1 for i, j in zip(m, n)])
    elif mode == 'valid':
        if not (all(m_d >= n_d for m_d, n_d in zip(m, n)) or
                all(m_d <= n_d for m_d, n_d in zip(m, n))):
            raise ValueError('In valid mode, either data or filter must be '
                             'at least as large as the other in every axis.')

        p = tuple([max(i, j) - min(i, j) + 1 for i, j in zip(m, n)])

    elif mode == 'same':
        p = tuple([i for i, j in zip(m, n)])
    else:
        raise ValueError(f'Invalid mode, got {mode}')

    return s, p


def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()
