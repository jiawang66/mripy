"""
Only for testing some ideas
"""

import sys
import numpy as np

sys.path.append('../../mripy')
import mripy.signal as sig


def main():

    xshape = (5, 4)
    shifts = ()
    axes = ()

    print('Circular shift.')
    x = np.array(range(np.prod(xshape))).reshape(xshape)
    y = sig.util.circshift(x, shifts=shifts, axes=axes)
    print('x = ', x)
    print('y = ', y)

    print()
    print('Linear shift.')
    x = np.array(range(np.prod(xshape))).reshape(xshape)
    y = sig.util.linshift(x, shifts=shifts, axes=axes, cval=0)
    print('x = \n', x)
    print('y = \n', y)

if __name__ == '__main__':
    main()
