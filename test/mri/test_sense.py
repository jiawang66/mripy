# -*- coding: utf-8 -*-
"""
Test SENSE
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
sys.path.append('../../../mripy')
from mripy.mri import coil_com, coil_smap, sample, sense
from mripy.signal import fourier, util


def display(arr, title='', cmap='gray', fill=0, grid_shape=None,
            caxis=None, padding_width=10, axis_on_off='off'):
    temp = util.montageaxis(arr, fill=fill, grid_shape=grid_shape,
                            padding_width=padding_width,
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

    # set simulation parameters
    caxis = 2
    img_axes = [0, 1]
    ishape = kdata.shape
    axis = 0  # acceleration axis
    rate = 4  # acceleration rate

    print(f'The acceleration axis is {axis}')
    print(f'The acceleration rate is {rate}')
    print(f'The channel axis is {caxis}')

    print('Reconstrunction of ground truth by IFFT ...', end='')
    img = fourier.ifftn(kdata, axes=img_axes)
    print('done.')

    # # Magnitude of img
    # img_mag = np.abs(img)
    # display(img_mag, title='Magnitude of ground truth', grid_shape=[2, 4], caxis=caxis)
    #
    # # Phase of img
    # img_ph = np.angle(img)
    # display(img_ph, title='Phase of ground truth', grid_shape=[2, 4], caxis=caxis)

    print('Coil combination...', end='')
    img_com = coil_com.sos(img, caxis=caxis)
    print('done.')
    display(img_com, title='Ground truth image combined by SOS', caxis=caxis, padding_width=0)

    print('Compute coil sensitivity map...', end='')
    smap = coil_smap.smap(img, img_com, caxis=caxis)
    print('done.')
    display(np.abs(smap), title='Smap', grid_shape=[2, 4], caxis=caxis)

    print('Generate uniform Cartesian sampling mask...', end='')
    mask = sample.uniform_cartesian(np.array(ishape)[img_axes], rate=rate,
                                    axis=axis if axis < caxis else axis - 1)
    print('done.')

    print('Downsampling the k-space data...', end='')
    kdata_samp = kdata * np.reshape(mask, [s if i != caxis else 1 for i, s in enumerate(ishape)])
    print('done.')

    print('Reconstrunction of wrapped image by IFFT ...', end='')
    img_wrapped = fourier.ifftn(kdata_samp, axes=img_axes)
    print('done.')

    # # Magnitude of img
    # img_wrapped_mag = np.abs(img_wrapped)
    # display(img_wrapped_mag, title='Magnitude of wrapped image', grid_shape=[2, 4], caxis=caxis)
    #
    # # Phase of img
    # img_wrapped_ph = np.angle(img_wrapped)
    # display(img_wrapped_ph, title='Phase of wrapped image', grid_shape=[2, 4], caxis=caxis)

    img_wrapped_com = coil_com.sos(img_wrapped, caxis=caxis)
    display(img_wrapped_com, title='Wrapped image combined by SOS', caxis=caxis, padding_width=0)

    print('SENSE reconstruction...')
    img_re, gfactor = sense.sense_1d(kdata_samp, smap, rate=rate, axis=axis, caxis=caxis)
    print('done.')
    display(np.abs(img_re), title='Mag of SENSE reconstruction', caxis=caxis, padding_width=0)
    display(np.abs(gfactor), title='gfactor', caxis=caxis, padding_width=0)

    print(np.max(np.abs(gfactor)))


if __name__ == '__main__':
    main()
