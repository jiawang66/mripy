"""
Only for testing some ideas
"""

import sys
import numpy as np

sys.path.append('../../mripy')
import mripy.signal as sig


def main():

    xshape = (3, 3)
    repeats = (2.0, 2)
    axes = (0, 1)

    x = np.array(range(np.prod(xshape))).reshape(xshape)
    y = sig.util.repeat(x, repeats, axes)

    print('x = \n', x)
    print('y = \n', y)
    print('ndim of x = ', x.ndim)

    yshape = y.shape
    xc = sig.util.arr_center(xshape)
    yc = sig.util.arr_center(yshape)
    shifts = [o - i for i, o in zip(xc, yc)]
    z = sig.util.circshift(y, shifts, axes)
    print('z = \n', z)

    wshape = [4, 1]
    w = sig.util.period_pad(x, wshape)
    print('w = \n', w)

if __name__ == '__main__':
    main()
