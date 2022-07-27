# -*- coding: utf-8 -*-
"""
Test convolution function
"""

import numpy as np
import scipy.signal as signal
import mripy.signal as sig


def vec(arr):
    return arr.reshape(-1)


def error(x0, y0, x1, y1):
    inner_1 = np.dot(vec(y0), vec(y1))
    inner_2 = np.dot(vec(x0), vec(x1))
    e12 = np.abs(inner_1 - inner_2) / np.abs(inner_1)
    print(f'error = {e12:.4e}')


def main():
    xshape = (9, )
    dshape = (4, )
    strides = (1, )
    modes = ('full', 'valid', 'same')

    x0 = np.random.rand(*xshape)
    d0 = np.random.rand(*dshape)

    for k, mode in enumerate(modes):
        print(f'mode = {mode}')

        print('conv and conv_data_adjoint')
        y0 = sig.conv.convolve(x0, d0, mode=mode, strides=strides)
        y1 = np.random.rand(*y0.shape)
        x1 = sig.conv.convolve_data_adjoint(y1, d0, data_shape=xshape, mode=mode, strides=strides)
        error(x0, y0, x1, y1)

        print('conv and conv_filt_adjoint')
        y0 = sig.conv.convolve(x0, d0, mode=mode, strides=strides)
        y1 = np.random.rand(*y0.shape)
        d1 = sig.conv.convolve_filter_adjoint(y1, x0, filt_shape=dshape, mode=mode, strides=strides)
        error(d0, y0, d1, y1)

        print()

if __name__ == '__main__':
    main()
