"""
Only for testing some ideas
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import mripy.mri as mri
import mripy.signal as sig


def display(arr, title='', cmap='gray', fill=0, grid_shape=None,
            channel_axis=None, axis_on_off='off'):
    temp = sig.util.montageaxis(arr, fill=fill, grid_shape=grid_shape,
                                channel_axis=channel_axis)
    plt.title(title)
    plt.imshow(temp, cmap=cmap)
    plt.axis(axis_on_off)
    plt.show()


def main():
    print('Load data...', end='')
    kdata = sio.loadmat('../data/kdata_8ch.mat')
    kdata = kdata['full_kspace_data']
    print('done.')

    print('Reconstrunction by IFFT ...', end='')
    img = sig.fourier.ifftn(kdata, axes=[0,1])
    print('done.')

    # Magnitude of img
    img_mag = np.abs(img)
    display(img_mag, title='Magnitude', grid_shape=[2, 4], channel_axis=2)

    # Phase of img
    img_ph = np.angle(img)
    display(img_ph, title='Magnitude', grid_shape=[2, 4], channel_axis=2)

    print('Coil combination...', end='')
    img_com = mri.coil_com.sos(img)
    print('done.')
    display(img_com, title='Image after coil combination by SOS')

    print('Compute coil sensitivity map...', end='')
    smap = mri.coil_smap.smap(img, img_com, 2)
    print('done.')
    display(np.abs(smap), title='Magnitude', grid_shape=[2, 4], channel_axis=2)


if __name__ == '__main__':
    main()
