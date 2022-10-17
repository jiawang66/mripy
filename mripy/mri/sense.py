# -*- coding: utf-8 -*-
"""SENSE reconstruction
Sensitivity encoding for fast MRI

"""

import warnings
import math
import numpy as np
from ..signal import fourier


def sense_1d(kdata, smap, rate, axis, caxis):
    """
    SENSE reconstruction for uniform downsampling along one axis.

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
    caxis : int
        Channel axis

    Returns
    -------
    img : ndarray
        Reconstructed image

    References
    ----------
    [1] Pruessmann K P, Weiger M, Scheidegger M B, et al. SENSE: sensitivity encoding
        for fast MRI[J]. Magnetic Resonance in Medicine, 1999, 42(5): 952-962.

    """
    ishape = kdata.shape
    ndim = len(ishape)
    nchannel = ishape[caxis]

    if not all([s1 == s2 for s1, s2 in zip(ishape, smap.shape)]):
        raise ValueError(f'Unmatch shape of kdata and smap, '
                         f'got {ishape} of kdata and {smap.shape} of smap')
    
    if rate < 1 or rate % 1 != 0:
        raise ValueError(f'rate must be a integer equal or larger than 1, got {rate}')
    
    if axis >= ndim:
        raise ValueError(f'Unknown axis of acceleration, got {axis}')
    
    if caxis >= ndim:
        raise ValueError(f'Unknown axis of channel, got {caxis}')
    
    if axis == caxis:
        raise ValueError(f'Acceleration axis cannot be same with the channel axis.')
    
    if rate > nchannel:
        warnings.warn(f'Acceleration rate of {rate} is larger than the size of channels of {nchannel}. '
                      f'The reconstructed results will be terrible.')

    if ishape[axis] % rate > 0:
        warnings.warn(f'The size of acceleration axis of {ishape[axis]} '
                      f'is not an integer multiple of the rate {rate}.')

    print('\n==========================================')
    print('Start SENSE Reconstruction. Please wait...')

    img_axes = list(range(ndim - 1))  # record the order of image axes
    img_axes.insert(caxis, -1)
    flag_swap = False
    if not (caxis == ndim - 1 or caxis == -1):
        flag_swap = True
        print(f'The channel axis is at dimension {caxis}. '
              f'Now put the channel axis to the last dimension...', end='')
        kdata = np.swapaxes(kdata, -1, caxis)
        smap = np.swapaxes(smap, -1, caxis)
        img_axes[-1], img_axes[caxis] = img_axes[caxis], img_axes[-1]
        if axis == ndim - 1 or axis == -1:
            axis = caxis
        print('done.')

    if not (axis == ndim - 2 or axis == -2):
        flag_swap = True
        print(f'The acceleration axis is at dimension {axis}. '
              f'Now put the acceleration axis to the second-to-last dimension...', end='')
        kdata = np.swapaxes(kdata, -2, axis)
        smap = np.swapaxes(smap, -2, axis)
        img_axes[-2], img_axes[axis] = img_axes[axis], img_axes[-2]
        print('done.')

    ishape = kdata.shape
    rate = int(rate)

    print('IFFT reconstruction of wrapped image...', end='')
    img_wrapped = fourier.ifftn(kdata, axes=list(range(ndim - 1)))
    print('done.')

    nwrap = int(ishape[-2] / rate)  # period in pixels of acceleration axis after wrap
    ny_floor = math.floor(ishape[-2] / rate) * rate

    print('Inverse reconstruction...', end='')
    img_re = np.zeros(ishape[:-1], dtype=kdata.dtype)
    for y in range(nwrap):
        slc = tuple([slice(0, None) if i != -2 else slice(y, y + 1) for i in range(-ndim, 0)])
        im = np.squeeze(img_wrapped[slc]).reshape([-1, nchannel])

        slc = tuple([slice(0, None) if i != -2 else slice(y, ny_floor, nwrap) for i in range(-ndim, 0)])
        sm = smap[slc].reshape([-1, rate, nchannel]).transpose([0, 2, 1])

        # More efficient way using np.einsum then explicit pinv along axis
        res = np.einsum('ijk,ik->ij', np.linalg.pinv(sm), im)
        img_re[slc[:-1]] = res.reshape(list(ishape)[:-2] + [rate, ])

    print('done')

    if flag_swap:
        print('Put all the axes back where the user had them...', end='')
        img_re = np.transpose(img_re, img_axes[:-1])
        print('done')

    print('Congratulation! SENSE reconstruction done.')
    print('==========================================\n')

    return img_re
