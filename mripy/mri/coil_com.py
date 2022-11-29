# -*- coding: utf-8 -*-
"""Coil combination
Combine all channels of MRI

"""

import numpy as np
from scipy.linalg import eigh
from skimage.filters import threshold_li

from ..signal import backend


def sos(img, caxis=None):
    """
    Coil combination by SOS (Sum of Squares)

    Parameters
    ----------
    img : ndarray
        multi-channel image, default the last dimension is channel
    caxis : None or int
        channel axis

    Returns
    -------
    image_after_combine : ndarray
    """
    shape = img.shape
    xp = backend.get_array_module(img)

    output = xp.sum(xp.abs(img)**2, axis=caxis) / shape[caxis]
    return xp.sqrt(output)


def adapt(img, caxis=-1, axes=None, mask=None):
    """
    Adaptive coil combination.

    Parameters
    ----------
    img : ndarray
        multi-channel image, default the last dimension is channel
    caxis : int
        channel axis
    axes: None or tuple or list of ints
        spatial axes of image, 2- or 3-element
    mask: None or ndarray
        A mask indicating which pixels of the coil sensitivity mask
        should be computed.

    Returns
    -------
    image_after_combine : ndarray
    smap : ndarray

    References
    ----------
    [1] Walsh et al. Adaptive Reconstruction of MRI array data,
        Magn Reson Med. 2000; 43(5):682-90

    """
    img = np.moveaxis(img, caxis, -1)
    nchannel = img.shape[-1]
    nvoxel = np.prod(img.shape[:-1])

    if mask is None:
        img_sos = sos(img, caxis=-1)
        thresh = threshold_li(img_sos)
        mask = (img_sos > thresh).flatten()
    else:
        mask = mask.flatten()

    assert mask.size == nvoxel, 'mask must be the same size as a coil.'

    # Compute the sample auto-covariances pointwise, will be
    # Hermitian symmetric, only need lower triangular matrix
    Rs = np.empty((nchannel, nchannel, nvoxel), dtype=img.dtype)
    for p in range(nchannel):
        for q in range(p):
            Rs[q, p, :] = (np.conj(img[..., p]) * img[..., q]).flatten()

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    csm = np.zeros((nvoxel, nchannel), dtype=img.dtype)
    for ii in np.nonzero(mask)[0]:
        R = Rs[..., ii]
        v = eigh(R, lower=False,
                 eigvals=(nchannel - 1, nchannel - 1))[1].squeeze()
        csm[ii, :] = v / np.linalg.norm(v)

    return np.moveaxis(np.reshape(csm, img.shape), -1, caxis)
