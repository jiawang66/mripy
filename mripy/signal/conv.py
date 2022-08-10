# -*- coding: utf-8 -*-
""" Convolution functions

Support N-dimension convolution with strides not equal to 1

"""

import warnings
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
from . import backend, util


__all__ = ['conv', 'conv_data_adjoint', 'conv_filt_adjoint']


def conv(data, filt, mode='full', strides=None):
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
    conv_data_adjoint
    conv_filt_adjoint

    """
    xp = backend.get_array_module(data)

    if xp == np:
        output = _conv(data, filt, mode=mode, strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(data)
        data = backend.to_device(data)
        filt = backend.to_device(filt)
        output = _conv(data, filt, mode=mode, strides=strides)
        output = backend.to_device(output, device)

    return output


def conv_data_adjoint(output, filt, data_shape, mode='full', strides=None):
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
    conv
    conv_filt_adjoint

    """
    data_shape = tuple(data_shape)
    xp = backend.get_array_module(output)

    if xp == np:
        data = _conv_data_adjoint(output, filt, data_shape,
                                  mode=mode, strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(output)
        output = backend.to_device(output)
        filt = backend.to_device(filt)
        data = _conv_data_adjoint(output, filt, data_shape,
                                  mode=mode, strides=strides)
        data = backend.to_device(data, device)

    return data


def conv_filt_adjoint(output, data, filt_shape, mode='full', strides=None):
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
    conv
    conv_data_adjoint

    """
    filt_shape = tuple(filt_shape)
    xp = backend.get_array_module(output)

    if xp == np:
        filt = _conv_filt_adjoint(output, data, filt_shape,
                                  mode=mode, strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(output)
        output = backend.to_device(output)
        data = backend.to_device(data)
        filt = _conv_filt_adjoint(output, data, filt_shape,
                                  mode=mode, strides=strides)
        filt = backend.to_device(filt, device)

    return filt


def cconv(data, filt, strides=None):
    """
    Circular convolution of two N-dimensional array.

    Parameters
    ----------
    data : ndarray
    filt : ndarray
    strides : list or tuple of ints, optional

    Returns
    -------

    """
    xp = backend.get_array_module(data)

    if xp == np:
        output = _cconv(data, filt, strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(data)
        data = backend.to_device(data)
        filt = backend.to_device(filt)
        output = _cconv(data, filt, strides=strides)
        output = backend.to_device(output, device)

    return output


def cconv_data_adjoint(output, filt, data_shape, strides=None):
    """
    Adjoint of circular convolution with respect to data

    Parameters
    ----------
    output : ndarray
    filt : ndarray
    data_shape
    strides

    Returns
    -------
    data : ndarray

    """
    data_shape = tuple(data_shape)
    xp = backend.get_array_module(output)

    if xp == np:
        data = _cconv_data_adjoint(output, filt, data_shape,
                                   strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(output)
        output = backend.to_device(output)
        filt = backend.to_device(filt)
        data = _cconv_data_adjoint(output, filt, data_shape,
                                   strides=strides)
        data = backend.to_device(data, device)

    return data


def cconv_filt_adjoint(output, data, filt_shape, strides=None):
    """
    Adjoint of circular convolution with respect to filter.

    Parameters
    ----------
    output : ndarray
    data : ndarray
    filt_shape
    strides

    Returns
    -------
    filt : ndarray

    """
    filt_shape = tuple(filt_shape)
    xp = backend.get_array_module(output)

    if xp == np:
        filt = _cconv_filt_adjoint(output, data, filt_shape,
                                   strides=strides)
    else:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(output)
        output = backend.to_device(output)
        data = backend.to_device(data)
        filt = _cconv_filt_adjoint(output, data, filt_shape,
                                   strides=strides)
        filt = backend.to_device(filt, device)

    return filt


def convmtx(data, filt_shape, mode='full', strides=None):
    """
    Construct a convolution matrix.

    Constructs the Toeplitz matrix representing N-dimensional convolution.

    The code `A = convmtx(data, filt_shape, mode)` creates a Toeplitz matrix A
    such that `A * filt` is equivalent to using `conv(data, filt, mode)`.
    The returned array A always has n columns. The number of rows depends on
    the specified mode, as explained below.

    Parameters
    ----------
    data : ndarray
        The N-dimensional array to convolve
    filt_shape : list or tuple
        The shape of filter. The number of columns in the resulting matrix A
        is `prod(filt_shape)`
    mode : str {`full`, `valid`, `same`}, optional
        A string indicating the size of the output. This is analogous to mode
        in numpy.convolve.
    strides : list or tuple of ints, None
        convolution strides of length of the number of dimension

    Returns
    -------
    A : ndarray
        The convolution matrix.

    See Also
    --------
    scipy.linalg.convolution_matrix

    """
    pass  # TODO


def _conv(data, filt, mode='full', strides=None):
    m, n, D, s, p = _get_conv_params(data.shape, filt.shape, mode, strides)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    output = signal.convolve(data, filt, mode=mode)[slc]
    return output


def _conv_data_adjoint(output, filt, data_shape,
                       mode='full', strides=None):
    m, n, D, s, p = _get_conv_params(data_shape, filt.shape, mode, strides)

    # if stride > 1, zero-fill the output
    if not s == (1,) * len(s):
        slc = tuple(slice(None, None, s_d) for s_d in s)
        output_fill = np.zeros(p, dtype=output.dtype)
        output_fill[slc] = output
        output = output_fill

    # zero-padding of the even-size dimensions to be odd-size
    n = tuple(k if k % 2 == 1 else k + 1 for k in n)
    filt = util.resize(filt, n)

    # reverse the filter
    filt = _reverse_and_conj(filt)

    if mode == 'full' or mode == 'valid':
        adjoint_mode = 'valid'

        # zero pad the output to have `full` size
        pad_shape = tuple([i + j - 1 for i, j in zip(m, n)])
        if not output.shape == pad_shape:
            # Pad on the left first.
            # That is, one more zero on the left than on the right
            num_left = [int(np.ceil((i - j) / 2)) for i, j in
                        zip(pad_shape, output.shape)]
            num_right = [(i - j) // 2 for i, j in
                         zip(pad_shape, output.shape)]
            output = util.crop_pad(output, dims=list(range(D)),
                                   num_left=num_left,
                                   num_right=num_right)
    else:
        adjoint_mode = 'same'

    data = signal.convolve(output, filt, mode=adjoint_mode)
    return data


def _conv_filt_adjoint(output, data, filt_shape,
                       mode='full', strides=None):
    m, n, D, s, p = _get_conv_params(data.shape, filt_shape, mode, strides)

    # if stride > 1, zero-fill the output
    # and then set the strides = None
    if not s == (1,) * len(s):
        slc = tuple(slice(None, None, s_d) for s_d in s)
        output_fill = np.zeros(p, dtype=output.dtype)
        output_fill[slc] = output
        output = output_fill
        strides = None

    if mode == 'full' or mode == 'valid':
        return _conv_data_adjoint(output, data, n,
                                  mode=mode, strides=strides)
    elif mode == 'same':
        adjoint_mode = 'full'
        flag = [True if i % 2 == 1 and j % 2 == 0 else False
                for i, j in zip(m, n)]
        output_shape = [i + j - 1if f else i
                        for i, j, f in zip(m, n, flag)]
        output = util.resize(output, output_shape)

        return _conv_data_adjoint(output, data, n,
                                  mode=adjoint_mode, strides=strides)
    else:
        raise ValueError(f'Invalid mode, got {mode}')


def _cconv(data, filt, strides=None):
    m, n, D, s, u = _get_cconv_params(data.shape, filt.shape, strides)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    filt = util.resize(filt, u)  # left zero-padding

    output = ndimage.convolve(data, filt, mode='wrap')[slc]
    return output


def _cconv_data_adjoint(output, filt, data_shape, strides=None):
    m, n, D, s, u = _get_cconv_params(data_shape, filt.shape, strides)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    filt = util.resize(filt, u)  # left zero-padding

    # if stride > 1, zero-fill the output
    if not s == (1,) * len(s):
        output_fill = np.zeros(m, dtype=output.dtype)
        output_fill[slc] = output
        output = output_fill

    # reverse the filter
    filt = _reverse_and_conj(filt)

    # adjoint
    data = ndimage.convolve(output, filt, mode='wrap')
    return data


def _cconv_filt_adjoint(output, data, filt_shape, strides=None):
    max_shape = [max(m_d, n_d) for m_d, n_d in zip(data.shape, filt_shape)]
    output  = util.resize(output, max_shape)
    data = util.resize(data, max_shape)
    filt = _cconv_data_adjoint(output, data, max_shape, strides=strides)
    return util.resize(filt, filt_shape)


def _get_conv_params(data_shape, filt_shape, mode, strides):
    """
    Check the convolution params and return the strides.
    """
    m = util.to_tuple(data_shape)
    n = util.to_tuple(filt_shape)
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

    return m, n, D, s, p


def _get_cconv_params(data_shape, filt_shape, strides):
    """
    Check the circular convolution params

    """
    m = util.to_tuple(data_shape)
    n = util.to_tuple(filt_shape)
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

    # If the size of one dimension of filter is even, then left pad 1 zero
    # filt_shape after zero-padding
    u = tuple(k if k % 2 == 1 else k + 1 for k in n)

    return m, n, D, s, u


def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()
