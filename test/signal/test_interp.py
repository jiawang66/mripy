# -*- coding: utf-8 -*-
"""
Test interpolation function
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

sys.path.append('../../../mripy')
from mripy.signal import interp


def test_zoom():
    ascent = misc.ascent()
    ishape = ascent.shape
    oshape = tuple([int(i * 0.5) for i in ishape])
    result = interp.zoom(ascent, oshape)

    plt.figure()
    plt.imshow(ascent, vmin=0, vmax=255, cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(result, vmin=0, vmax=255, cmap='gray')
    plt.show()

    print('ishape = ', ishape)
    print('oshape = ', oshape)
    print('rshape = ', result.shape)


def main():
    test_zoom()


if __name__ == '__main__':
    main()
