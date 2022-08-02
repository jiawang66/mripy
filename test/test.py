"""
Only for testing some ideas
"""

import sys
import numpy as np

sys.path.append('../../mripy')
import mripy.signal as sig


def main():

    xshape = (5, )
    dshape = (2, )
    strides = (2, )
    mode = 'valid'

    x = np.array(range(np.prod(xshape))).reshape(xshape)
    d = np.array(range(np.prod(dshape))).reshape(dshape)
    y = sig.conv.convolve(x, d, mode=mode, strides=strides)

    print('x = ', x)
    print('d = ', d)
    print('y = ', y)

if __name__ == '__main__':
    main()
