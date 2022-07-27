"""
Only for testing some ideas
"""

import numpy as np
import scipy.signal as signal
import mripy.signal as sig


EPS = 1e-16


def error(x0, y0, x1, y1):
    inner_1 = np.dot(vec(y0), vec(y1))
    inner_2 = np.dot(vec(x0), vec(x1))
    e12 = np.abs(inner_1 - inner_2) / np.abs(inner_1 + EPS)
    print(f'error = {e12:.4e}')

    assert e12 < 1e-12


def vec(arr):
    return arr.reshape(-1)


def run_test(xshape, dshape, mode):
    # x0 = np.random.randn(*xshape)
    # d0 = np.random.randn(*dshape)
    x0 = np.array(range(np.prod(xshape))).reshape(xshape)
    d0 = np.array(range(np.prod(dshape))).reshape(dshape)

    # convolution
    y0 = signal.convolve(x0, d0, mode=mode)
    y1 = np.random.randn(*y0.shape)
    # y1 = y0[::-1]

    # convolve_data_adjoint
    x1 = sig.conv.convolve_data_adjoint(y1, d0, xshape, mode=mode)
    error(x0, y0, x1, y1)

    # convolve_filt_adjoint
    d1 = sig.conv.convolve_filt_adjoint(y1, x0, dshape, mode=mode)
    error(d0, y0, d1, y1)


def main():

    xshape = (2, 3, 4, 3, 1, 1, 4, 2, 3)
    dshape = (3, 4, 2, 3, 4, 1, 1, 2, 3)
    mode = 'full'
    run_test(xshape, dshape, mode)

    mode = 'same'
    run_test(xshape, dshape, mode)

    xshape = (1, 2, 3, 2, 3, 3, 1, 2)
    dshape = (3, 3, 5, 4, 5, 3, 1, 2)
    mode = 'valid'
    run_test(xshape, dshape, mode)

    xshape, dshape = dshape, xshape
    run_test(xshape, dshape, mode)

if __name__ == '__main__':
    main()
