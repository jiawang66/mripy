# -*- coding: utf-8 -*-
"""Utility functions.

"""

import math
import numpy as np
import warnings
from collections import Iterable
from . import backend
from skimage.util import montage


def arr_center(shape):
    """Center of array

    Return the subscript of the center, so plus 1 for the real center.

    center = ceil(shape / 2)

    Note that for even-size shape, the center is on the left side
    of ceil(shape / 2), which is different with MATLAB.

    Args:
        shape (int or list or tuple or ndarray): shape of an array or vector

    Returns:
        list: center of array

    """
    check_shape_positive(shape)
    return list([math.ceil(i / 2) - 1 for i in shape])


def check_shape_positive(shape, subject=None):
    if subject is None:
        subject = 'Shapes'

    if not isinstance(shape, tuple) and not isinstance(shape, list):
        shape = (shape, )

    if not all(s > 0 for s in shape):
        raise ValueError(f'{subject} must be positive, got {shape}')

    if not all(s % 1 == 0 for s in shape):
        raise ValueError(f'{subject} must be integers, got {shape}')


def expand_shapes(*shapes):
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)


def normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))


def prod(shape):
    """Computes product of shape.

    Args:
        shape (tuple or list): shape.

    Returns:
        Product.

    """
    return np.prod(shape, dtype=np.int64)


def montageaxis(arr_in, channel_axis=None, fill='mean', rescale_intensity=False,
                grid_shape=None, padding_width=0, multichannel=False):
    """
    Create a montage of several single- or multichannel images
    with specific channel_axis

    Args:
        arr_in (ndarray): (K, M, N[, C]) ndarray,
        an array representing an ensemble of `K` images of equal shape.
        channel_axis (None or int): channel axis

    See Also:
        :func:`skimage.util.montage`
    """
    arr_in = backend.to_device(arr_in)  # move to cpu device

    if channel_axis is not None and not arr_in.ndim == 2:
        ndim = arr_in.ndim
        axis = normalize_axes((channel_axis,), ndim)[0]
        axes = list(range(ndim))
        axes.remove(axis)
        axes.insert(0, axis)
        arr_in = np.transpose(arr_in, axes=axes)

    if arr_in.ndim == 2:
        arr_in = np.expand_dims(arr_in, 0)

    arr_out = montage(arr_in, fill=fill, rescale_intensity=rescale_intensity,
                      grid_shape=grid_shape, padding_width=padding_width,
                      multichannel=multichannel)
    return arr_out


def crop_pad(arr_in, dims, num_left=None, num_right=None, cval=0):
    """
    Crop or pad an array

    Parameters
    ----------
    arr_in : ndarray
        The array to pad or crop
    dims : int or list or tuple
        The dimensions of length-N that be applied crop or pad
    num_left : int or list or tuple or None
        Length-N vector, the pixels crop or pad at left side:
        positive : pad,
        negative : crop.
    num_right : int or list or tuple or None
        Length-N vector, the pixels crop or pad at right side:
        positive : pad,
        negative : crop.
    cval : int or float or None
        The value to set the padded values

    Returns
    -------
    arr_out : ndarray
    """
    flag_1 = not num_left
    flag_2 = not num_right
    if flag_1 and flag_2:
        return arr_in
    elif flag_1:
        num_left = [0, ] * np.size(num_right)
    elif flag_2:
        num_right = [0, ] * np.size(num_left)

    if not isinstance(dims, Iterable):
        dims = (dims, )
    if not isinstance(num_left, Iterable):
        num_left = (num_left, )
    if not isinstance(num_right, Iterable):
        num_right = (num_right, )

    if len(dims) != len(num_left) or len(dims) != len(num_right):
        raise ValueError(f'Cannot determine crop or pad dimension, '
                         f'got dims {dims}, '
                         f'got num_left {num_left}, '
                         f'got num_right {num_right}.')

    ndim = arr_in.ndim
    num_left = [num_left[dims.index(k)] if k in dims else 0 for k in range(ndim)]
    num_right = [num_right[dims.index(k)] if k in dims else 0 for k in range(ndim)]

    ishape = arr_in.shape
    oshape = tuple(ni + nl + nr for ni, nl, nr in zip(ishape, num_left, num_right))

    istart = [-nl if nl < 0 else 0 for nl in num_left]
    ostart = [nl if nl > 0 else 0 for nl in num_left]
    iend = [nr if nr < 0 else ni for nr, ni in zip(num_right, ishape)]
    oend = [-nr if nr > 0 else no for nr, no in zip(num_right, oshape)]
    islice = tuple([slice(start, end) for start, end in zip(istart, iend)])
    oslice = tuple([slice(start, end) for start, end in zip(ostart, oend)])

    xp = backend.get_array_module(arr_in)
    arr_out = xp.ones(oshape, dtype=arr_in.dtype) * cval
    arr_out[oslice] = arr_in[islice]

    return arr_out


def resize(arr_in, oshape, ishift=None, oshift=None, cval=0):
    """Resize with padding or cropping around center.

    If want to zero-padding or cropping around center, then ishift and oshift should be None
    and the resize function will compute them automatically.

    Args:
        arr_in (ndarray): Input array.
        oshape (tuple or list of ints): Output shape.
        ishift (None or list or tuple of ints): Input shift.
        oshift (None or list or tuple of ints): Output shift.
        constant_values (None or ints or float): The values to set the padded values for all axes.

    Returns:
        ndarray: Zero-padded or cropped result.
    """

    ishape1, oshape1 = expand_shapes(arr_in.shape, oshape)

    if ishape1 == oshape1:
        return arr_in.reshape(oshape)

    ic = arr_center(ishape1)
    oc = arr_center(oshape1)

    if ishift is None:
        ishift = [max(i - o, 0) for i, o in zip(ic, oc)]  # start index of arr_in

    if oshift is None:
        oshift = [max(o - i, 0) for i, o in zip(ic, oc)]  # start index of arr_out

    copy_shape = [min(i - si, o - so) for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    xp = backend.get_array_module(arr_in)
    arr_out = xp.ones(oshape1, dtype=arr_in.dtype) * cval
    arr_in = arr_in.reshape(ishape1)
    arr_out[oslice] = arr_in[islice]

    return arr_out.reshape(oshape)


def circshift(arr_in, shifts, axes=None):
    """
    Circular shift of `arr_in`

    Parameters
    ----------
    arr_in : ndarray
        input data
    shifts : list or tuple of ints
        shifts
    axes : list or tuple of ints, optional
        Axes to perform circshift operation

    Returns
    -------
    ndarray

    """
    if axes is None:
        axes = list(range(arr_in.ndim))

    axes = to_tuple(axes)
    shifts = to_tuple(shifts)
    assert(len(axes) == len(shifts))
    xp = backend.get_array_module(arr_in)

    for shift, axis in zip(shifts, axes):
        arr_in = xp.roll(arr_in, shift, axis=axis)

    return arr_in


def linshift(arr_in, shifts, axes=None, cval=0.0):
    """
    Linear shift of `arr_in`

    Parameters
    ----------
    arr_in : ndarray
        input data
    shifts : list or tuple of ints
        shifts
    axes : list or tuple of ints, optional
        Axes to perform circshift operation
    cval : float
        constant value to fill past edges of arr_in. Default 0.

    Returns
    -------
    ndarray

    """
    if axes is None:
        axes = list(range(arr_in.ndim))

    axes = to_tuple(axes)
    shifts = to_tuple(shifts)
    assert (len(axes) == len(shifts))
    xp = backend.get_array_module(arr_in)

    if not xp == np:
        warnings.warn('Currently not support convolution of cupy.ndarray. '
                      'Try to transform it to numpy.ndarray.')
        device = backend.get_device(arr_in)
        arr_in = backend.to_device(arr_in)
    else:
        device = backend.cpu_device

    for shift, axis in zip(shifts, axes):
        arr_in = _linshift(arr_in, shift, axis, cval=cval)

    if not xp == np:
        arr_in = backend.to_device(arr_in, device)

    return arr_in


def to_list(a, order=None):
    """
    Convert array_like input to a list

    Parameters
    ----------
    a : array_like
        input data, in any form that can be converted to an array.
    order : {'C', 'F'}, optional
        'C' : row-major (C-style) (default)
        'F' : column-major (Fortran-style)

    Returns
    -------
    out : list

    """
    return list(vec(a, order=order))


def to_tuple(a, order=None):
    """
    Convert array_like input to a tuple

    Parameters
    ----------
    a : array_like
        input data, in any form that can be converted to an array.
    order : {'C', 'F'}, optional
        'C' : row-major (C-style) (default)
        'F' : column-major (Fortran-style)

    Returns
    -------
    out : tuple

    """
    return tuple(vec(a, order=order))


def vec(a, order=None, column=False):
    """
    Convert the input to a 1-D vector

    Parameters
    ----------
    a : array_like
        input data, in any form that can be converted to an array.
    order : {'C', 'F'}, optional
        'C' : row-major (C-style) (default)
        'F' : column-major (Fortran-style)
    column : bool
        if true, return a column vector

    Returns
    -------
    out : ndarray
    """
    b = np.asarray(a)

    if column:
        return b.reshape((-1, 1), order=order)
    else:
        return b.reshape(-1, order=order)


def _linshift(arr_in, shift, axis, cval=0.0):
    if shift % 1 != 0:
        raise ValueError(f'Shift must be an integer, got {shift}.')

    ndim = arr_in.ndim
    if axis >= ndim:
        raise ValueError(f'axis {axis} is out of bounds for array of dimension {ndim}.')

    shape = arr_in.shape
    istart, ostart = [0, ] * ndim, [0, ] * ndim
    iend, oend = list(shape), list(shape)

    if shift > 0:
        iend[axis] = -shift
        ostart[axis] = shift
    elif shift < 0:
        istart[axis] = -shift
        oend[axis] = shift

    islice = tuple([slice(start, end) for start, end in zip(istart, iend)])
    oslice = tuple([slice(start, end) for start, end in zip(ostart, oend)])

    arr_out = np.ones(shape, dtype=arr_in.dtype) * cval
    arr_out[oslice] = arr_in[islice]

    return arr_out
