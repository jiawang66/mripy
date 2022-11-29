# -*- coding: utf-8 -*-
"""
Test sensitivity map function
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
sys.path.append('../../../mripy')
from mripy.mri import coil_com, coil_smap
from mripy.signal import fourier, util


def display(arr, title='', cmap='gray', fill=0, grid_shape=None,
            caxis=None, axis_on_off='off'):
    temp = util.montageaxis(arr, fill=fill, grid_shape=grid_shape,
                                caxis=caxis)

    plt.figure(figsize=[8,5])
    plt.title(title, fontsize=18)
    plt.imshow(temp, cmap=cmap)
    plt.axis(axis_on_off)
    plt.show()


def main():
    print('Load data...', end='')
    kdata = sio.loadmat('../../data/kdata_8ch.mat')
    kdata = kdata['full_kspace_data']
    print('done.')

    print('Reconstrunction by IFFT ...', end='')
    img = fourier.ifftn(kdata, axes=[0,1])
    print('done.')

    # Magnitude of img
    img_mag = np.abs(img)
    display(img_mag, title='Magnitude', grid_shape=[2, 4], caxis=2)

    # Phase of img
    img_ph = np.angle(img)
    display(img_ph, title='Phase', grid_shape=[2, 4], caxis=2)

    print('Coil combination...', end='')
    img_com = coil_com.sos(img)
    print('done.')
    display(img_com, title='Image combined by SOS', caxis=2)

    print('Compute coil sensitivity map...', end='')
    smap = coil_smap.smap(img, img_com, 2)
    print('done.')
    display(np.abs(smap), title='Smap', grid_shape=[2, 4], caxis=2)


if __name__ == '__main__':
    main()
