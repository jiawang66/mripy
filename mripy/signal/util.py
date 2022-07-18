# -*- coding: utf-8 -*-
"""Utility functions.

"""

import math
import numpy as np
from . import backend


def arr_center(shape):
    """Center of array

    center = ceil(shape / 2)

    Note that for even-size shape, the center is on the left side
    of ceil(shape / 2), which is different with MATLAB.

    Args:
        shape (int or list or tuple or ndarray): shape of an array or vector

    Returns:
        list: center of array

    """
    check_shape_positive(shape)
    return [math.ceil(i / 2) - 1 for i in shape]


def check_shape_positive(shape):
    if not all(s > 0 for s in shape):
        raise ValueError(f'Shapes must be positive, got {shape}')

    if not all(s % 1 == 0 for s in shape):
        raise ValueError(f'Shapes must be integers, got {shape}')


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


def resize(input, oshape, ishift=None, oshift=None, constant_value=0):
    """Resize with zero-padding or cropping around center.

    If want to zero-padding or cropping around center, then ishift and oshift should be None
    and the resize function will compute them automatically.

    Args:
        input (ndarray): Input array.
        oshape (tuple or list of ints): Output shape.
        ishift (None or list or tuple of ints): Input shift.
        oshift (None or list or tuple of ints): Output shift.
        constant_values (None or ints or float): The values to set the padded values for all axes.

    Returns:
        ndarray: Zero-padded or cropped result.
    """

    ishape1, oshape1 = expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    ic = arr_center(ishape1)
    oc = arr_center(oshape1)

    if ishift is None:
        ishift = [max(i - o, 0) for i, o in zip(ic, oc)]  # start index of input

    if oshift is None:
        oshift = [max(o - i, 0) for i, o in zip(ic, oc)]  # start index of output

    copy_shape = [min(i - si, o - so) for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    xp = backend.get_array_module(input)
    output = xp.ones(oshape1, dtype=input.dtype) * constant_value
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output.reshape(oshape)
