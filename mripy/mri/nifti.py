# -*- coding: utf-8 -*-
"""
NIFTI image

"""

import os
import nibabel as nib
import numpy as np
import mripy.signal as sig


def load_nii(filename, **kwargs):
    """
    Load file given filename, guessing at file type

    """
    nii = nib.loadsave.load(filename, **kwargs)
    return nii


def save_nii(nii, filename):
    """
    Save an image to file adapting format to `filename`

    """
    nib.loadsave.save(nii, filename)


def make_nii(img, voxel_size=None, affine=None, header=None):
    """
    Make NIFTI based on nibabel.Nifti1Image

    Parameters
    ----------
    img : ndarray
        image data
    voxel_size : float, list, tuple, optional
        voxel size of the image, in millimeter
    affine : ndarray, optional
        4 x 4 affine matrix
    header : nibabel.Nifti1Header, optional
        header of nifti

    Returns
    -------
    nii : nibabel.Nifti1Image

    Notes
    -----
    If both a `voxel_size` and an `affine` are specified, a new affine matrix
    will be computed based on that the image center have the same world
    coordinate.

    If both a `header` and an `affine` are specified, and the `affine` does
    not match the affine that is in the `header`, the `affine` will be used,
    but the ``sform_code`` and ``qform_code`` fields in the header will be
    re-initialised to their default values.

    """

    if img.ndim > 7:
        raise ValueError(f'Number of dimension of `img` '
                         f'must be less than 8, '
                         f'got shape {img.shape}.')

    if voxel_size is None and affine is None and header is None:
        voxel_size = (1, ) * 3

    if voxel_size is not None:
        voxel_size = sig.util.to_tuple(voxel_size)
        if len(voxel_size) > 3:
            raise ValueError(f'Length of voxel_size must be less than 4. '
                             f'Got {voxel_size}.')
        voxel_size = voxel_size + (1, ) * (3 - len(voxel_size))

        if affine is None and header is None:
            affine = np.eye(4)
            affine[[0,1,2],[0,1,2]] = voxel_size
        elif affine is not None:
            affine = _new_affine(img.shape, affine, img.shape, voxel_size)
        elif header is not None:
            affine = header.get_best_affine()
            affine = _new_affine(img.shape, affine, img.shape, voxel_size)

    nii = nib.Nifti1Image(img, affine, header=header)
    nii.header.set_dim_info()
    # nii.header._structarr['pixdim'][1:4] = voxel_size

    return nii


def crop_pad(nii, num_left=None, num_right=None):
    pass


def resize(nii, shape_new=None):
    pass


def _new_affine(shape_old, affine_old, shape_new, voxel_size_new=None):
    """
    Compute new affine matrix. Note that the central voxel
    in the old and new image have the same world coordinate.

    Parameters
    ----------
    shape_old : list or tuple
        shape of old image
    affine_old : ndarray
        4 x 4 affine matrix
    shape_new : list or tuple
        shape of new image
    voxel_size_new : list or tuple, optional
        voxel_size of the new image. If not provided,
        it will be computed based on the assumption that
        the new and the old image have the same FOV.

    Returns
    -------
    affine_new : ndarray
        4 x 4 affine matrix
    """
    shape_old = sig.util.to_tuple(shape_old)
    shape_new = sig.util.to_tuple(shape_new)

    if shape_new == shape_old and voxel_size_new is None:
        return affine_old

    if voxel_size_new is None:
        voxel_size_old = affine_old[[0, 1, 2], [0, 1, 2]]
        voxel_size_new = np.array(shape_old) * voxel_size_old \
                         / np.array(shape_new)

    voxel_size_new = sig.util.to_tuple(voxel_size_new)

    _center = sig.util.arr_center
    _vec = sig.util.vec

    c_old = _center(shape_old)[:3]
    c_old = _vec(list(c_old) + [0, ] * (3 - len(c_old)), column=True)
    c_new = _center(shape_new)[:3]
    c_new = _vec(list(c_new) + [0, ] * (3 - len(c_new)), column=True)
    c_world_old = np.dot(affine_old[:3, :3], c_old) \
                  + _vec(affine_old[:3, 3], column=True)
    affine_new = np.array(affine_old)
    affine_new[[0, 1, 2], [0, 1, 2]] = voxel_size_new
    affine_new[:3, 3] = _vec(c_world_old - np.dot(affine_new[:3, :3], c_new))

    return affine_new
