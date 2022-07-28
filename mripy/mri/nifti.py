# -*- coding: utf-8 -*-
"""
NIFTI image

"""

import os
import nibabel as nib
import numpy as np
from mripy.signal import util


def load_nii(filename, **kwargs):
    nii = nib.loadsave.load(filename, **kwargs)
    return nii

def save_nii(nii, filename):
    nib.loadsave.save(nii, filename)


def make_nii(img, voxel_size=None, affine=None, header=None):
    """
    Make NIFTI based on Nifti1Image

    Parameters
    ----------
    img : ndarray
    voxel_size : float, list, tuple, None

    Returns
    -------
    nii : Nifti1Image

    """

    if img.ndim > 7:
        raise ValueError(f'Number of dimension of `img` '
                         f'must be less than 8, '
                         f'got shape {img.shape}.')

    if voxel_size is None and affine is None and header is None:
        voxel_size = (1, ) * 3

    if voxel_size is not None:
        if np.isscalar(voxel_size):
            voxel_size = (voxel_size, ) * 3
        else:
            if len(voxel_size) > 3:
                raise ValueError(f'Length of voxel_size must be less than 4. '
                                 f'Got {voxel_size}.')
            voxel_size = tuple(voxel_size) + (1, ) * (3 - len(voxel_size))

        if affine is None:
            affine = np.eye(4)
            affine[[0,1,2],[0,1,2]] = voxel_size
        else:
            center = util.arr_center(img.shape)[:3]
            center = np.asarray(list(center) + [0, ] * (3 - len(center))).reshape(3, 1)
            center_world = np.dot(affine[:3, :3], center) + np.asarray(affine[:3, 3]).reshape(3, 1)
            affine[[0,1,2],[0,1,2]] = voxel_size
            affine[:3, 3] = (center_world - np.dot(affine[:3, :3], center)).reshape(3,)

    nii = nib.Nifti1Image(img, affine, header=header)
    nii.header.set_dim_info()
    # nii.header._structarr['pixdim'][1:4] = voxel_size

    return nii