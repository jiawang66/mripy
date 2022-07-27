"""
Only for testing some ideas
"""

import numpy as np
import mripy.signal as sig
from scipy.signal import convolve, correlate


def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()


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
    filt = sig.util.resize(filt, filt_shape)
    filt = _reverse_and_conj(filt)

    if mode == 'full' or mode == 'valid':
        adjoint_mode = 'valid'

        # zero pad the output
        pad_shape = [i + j - 1 for i, j in zip(data_shape, filt_shape)]
        num_left = [int(np.ceil((i - j) / 2)) for i, j in zip(pad_shape, output.shape)]
        num_right = [(i - j) // 2 for i, j in zip(pad_shape, output.shape)]
        dims = list(range(output.ndim))
        output = sig.util.crop_pad(output, dims=dims, num_left=num_left, num_right=num_right)

    elif mode == 'same':
        adjoint_mode = 'same'
    else:
        raise ValueError(f'Unknown mode, got {mode}')

    data = convolve(output, filt, mode=adjoint_mode)
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
    return convolve_data_adjoint(output, data, filt_shape, mode=mode)


def error(x0, y0, x1, y1):
    inner_1 = np.dot(vec(y0), vec(y1))
    inner_2 = np.dot(vec(x0), vec(x1))
    e12 = np.abs(inner_1 - inner_2) / np.abs(inner_1)
    print(f'error = {e12:.4e}')

    assert e12 < 1e-12


def vec(arr):
    return arr.reshape(-1)


def main():
    xshape = (1, 6, 7, 8)
    dshape = (8, 7, 1, 6)
    # xshape = (2, 3, 3, 2)
    # dshape = (5, 6, 7, 8)
    mode = 'same'

    # x0 = np.random.randn(*xshape)
    # d0 = np.random.randn(*dshape)
    x0 = np.array(range(np.prod(xshape))).reshape(xshape)
    d0 = np.array(range(np.prod(dshape))).reshape(dshape)

    # convolution
    y0 = convolve(x0, d0, mode=mode)
    y1 = np.random.randn(*y0.shape)

    # convolve_data_adjoint
    x1 = convolve_data_adjoint(y1, d0, xshape, mode=mode)
    error(x0, y0, x1, y1)

    # convolve_filt_adjoint
    d1 = convolve_filt_adjoint(y0, x0, dshape, mode=mode)
    error(d0, y0, d1, y1)


if __name__ == '__main__':
    main()
