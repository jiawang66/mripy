# -*- coding: utf-8 -*-
"""SENSE reconstruction
Sensitivity encoding for fast MRI

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from ..signal import fourier
from ..signal import util as sig_util


def _display(arr, title='', cmap='gray', fill=0, grid_shape=None,
            channel_axis=None, padding_width=10, axis_on_off='off'):
    temp = sig_util.montageaxis(arr, fill=fill, grid_shape=grid_shape,
                                padding_width=padding_width,
                                channel_axis=channel_axis)

    plt.figure(figsize=[8,5])
    plt.title(title, fontsize=18)
    plt.imshow(temp, cmap=cmap)
    plt.axis(axis_on_off)
    plt.show()


def sense(kdata, smap, rate, axis, channel_axis):
    """
    SENSE reconstruction of uniform downsampling image.

    Parameters
    ----------
    kdata : ndarray
        Downsampled k-sapce data. Non-acquired points should be filled with zero.
    smap : ndarray
        Coil sensitivity map
    rate: int
        Acceleration rate, should be equal or larger than 1
    axis : int
        Acceleration dimension
    channel_axis : int
        Channel axis

    Returns
    -------
    img : ndarray
        Reconstructed image

    References
    ----------
    [1] Pruessmann K P, Weiger M, Scheidegger M B, et al. SENSE: sensitivity encoding
        for fast MRI[J]. Magnetic Resonance in Medicine, 1999, 42(5): 952-962.

    TODO
    ----
    [1] Raise warning
    [2] Fail when the size of acceleration axis is not an integer multiple of the rate.
    [3] Loss the phase? Not sure. What if the ground truth is in real value?

    """
    ishape = kdata.shape
    ndim = len(ishape)
    nchannel = ishape[channel_axis]

    if not all([s1 == s2 for s1, s2 in zip(ishape, smap.shape)]):
        raise ValueError(f'Unmatch shape of kdata and smap, '
                         f'got {ishape} of kdata and {smap.shape} of smap')
    
    if rate < 1 or rate % 1 != 0:
        raise ValueError(f'rate must be a integer equal or larger than 1, got {rate}')
    
    if axis >= ndim:
        raise ValueError(f'Unknown axis of acceleration, got {axis}')
    
    if channel_axis >= ndim:
        raise ValueError(f'Unknown axis of channel, got {channel_axis}')
    
    if axis == channel_axis:
        raise ValueError(f'Acceleration axis cannot be same with the channel axis.')
    
    if rate > nchannel:
        Warning(f'Acceleration rate of {rate} is larger than the size of channels {nchannel}. '
                f'The reconstructed results will be terrible.')

    axis_orig = axis
    channel_axis_orig = channel_axis
    if channel_axis != 0:
        print(f'The channel axis is at dimension {channel_axis}. '
              f'Now put the channel axis to first dimension...', end='')
        kdata = np.swapaxes(kdata, 0, channel_axis)
        smap = np.swapaxes(smap, 0, channel_axis)
        ishape = kdata.shape
        if axis == 0:
            axis = channel_axis
        channel_axis = 0
        print('done.')

    if axis != 1:
        print(f'The acceleration axis is at dimension {axis}. '
              f'Now put the acceleration axis to second dimension...', end='')
        kdata = np.swapaxes(kdata, 1, axis)
        smap = np.swapaxes(smap, 1, axis)
        ishape = kdata.shape
        axis = 1
        print('done.')

    print('Start SENSE Reconstruction. Please wait...')

    flag = False
    tshape = list(ishape)
    tslice = tuple([slice(0, s) for s in ishape])
    if ishape[axis] % rate != 0:
        print('Pad the kdata along acceleration axis...', end='')
        flag = True
        tshape[axis] = math.ceil(tshape[axis] / rate) * rate
        kdata_pad = np.zeros(tshape, dtype=kdata.dtype)
        kdata_pad[tslice] = kdata
        kdata = kdata_pad
        del kdata_pad

        smap_pad = np.zeros(tshape, dtype=kdata.dtype)
        smap_pad[tslice] = fourier.fftn(smap, axes=axis)  # smap in k-space
        smap = fourier.ifftn(smap_pad, axes=axis)  # smap in image domain, but padded
        del smap_pad
        print('done.')

    print('IFFT reconstruction of wrapped image...', end='')
    img_wrapped = fourier.ifftn(kdata, axes=[i for i in range(ndim) if i != channel_axis])
    print('done.')

    # TEST
    # _display(np.abs(img_wrapped), title='TEST', grid_shape=[2, 4], channel_axis=channel_axis)
    # END TEST

    nwrap = int(tshape[axis] / rate)  # period in pixels of acceleration axis after wrap

    print('Inverse reconstruction...', end='')
    img_re = np.zeros(tshape[1:], dtype=kdata.dtype)
    for y in range(nwrap):
        tslc = tuple([slice(0, None) if i != 1 else slice(y, y + 1) for i in range(ndim)])
        img_slice = np.squeeze(img_wrapped[tslc])
        img_slice = img_slice.reshape([nchannel, -1])

        tslc = tuple([slice(0, None) if i != 1 else slice(y, None, nwrap) for i in range(ndim)])
        smap_slice = smap[tslc].reshape([nchannel, rate, -1])

        temp = np.zeros([rate, img_slice.shape[1]], dtype=kdata.dtype)
        for i in range(img_slice.shape[1]):
            I = sig_util.vec(img_slice[:, i], column=True)
            C = np.squeeze(smap_slice[:, :, i])
            # temp[:, i] = sig_util.vec(np.linalg.pinv(C) @ I)
            m = np.linalg.pinv(C.transpose() @ C) @ C.transpose() @ I
            temp[:, i] = sig_util.vec(m)

        tslc = tuple([slice(0, None) if i != 0 else slice(y, None, nwrap) for i in range(ndim - 1)])
        img_re[tslc] = temp.reshape([rate, ] + tshape[2:])
    
    # if flag:
    #     img_re_kspace = fourier.fftn(img_re, axes=list(range(ndim - 1)))
    #     img_re_kspace = img_re_kspace[tslice[1:]]
    #     img_re = fourier.ifftn(img_re_kspace, axes=list(range(ndim - 1)))
    print('SENSE reconstruction done.')

    return img_re
