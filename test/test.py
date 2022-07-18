"""
Only for testing some ideas
"""

import numpy as np
import mripy.signal as sig

def main():
    x1 = np.array([1,2,-3,4,5])
    y1 = np.array([1,2,-2])

    print('x1 = ', x1)
    print('y1 = ', y1)

    z1 = np.convolve(x1, y1, 'same')
    print('z1 = ', z1)

    y2 = sig.util.resize(y1, [4])
    print('y2 = ', y2)
    z2 = np.convolve(x1, y2, 'same')
    print('z2 = ', z2)

    shape = [1, 2, 3, 4, 5]
    print('center = ', sig.util.arr_center(shape))

    x = np.asarray(range(9))
    y = sig.util.resize(x, [10])
    print(x)
    print(y)


if __name__ == '__main__':
    main()
