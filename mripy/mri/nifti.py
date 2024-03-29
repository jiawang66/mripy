# -*- coding: utf-8 -*-
"""
NIFTI image

"""

import os
import platform
import nibabel as nib
import numpy as np
from ..signal import util as sig_util
from ..signal import interp


def load_nii(filename, **kwargs):
    """
    Load file given filename, guessing at file type

    Returns
    -------
    img : ``SpatialImage``
       Image of guessed type

    """
    nii = nib.loadsave.load(filename, **kwargs)
    return nii


def save_nii(nii, filename):
    """
    Save an image to file adapting format to `filename`

    """
    nib.loadsave.save(nii, filename)


def save_nii_quick(img, filename, header=None):
    """
    Save an image with a referenced nifti `header`

    Parameters
    ----------
    img : ndarray
        array image to save
    filename : str
        save filename
    header : nibabel.Nifti1Header, optional
        reference nifti header

    """
    # TODO : voxel size
    nii = make_nii(img, header=header)
    save_nii(nii, filename)


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
        voxel_size = sig_util.to_tuple(voxel_size)
        if len(voxel_size) > 7:
            raise ValueError(f'Length of voxel_size must be less than 8. '
                             f'Got {voxel_size}.')
        voxel_size = voxel_size + (1, ) * (3 - len(voxel_size))

        if affine is None and header is None:
            affine = np.eye(4)
            affine[[0,1,2],[0,1,2]] = voxel_size[:3]
        elif affine is not None:
            affine = _new_affine(img.shape, affine, img.shape, voxel_size[:3])
        elif header is not None:
            affine = header.get_best_affine()
            affine = _new_affine(img.shape, affine, img.shape, voxel_size[:3])
    else:
        voxel_size = ()

    nii = nib.Nifti1Image(img, affine, header=header)
    nii.header._structarr['pixdim'][1:len(voxel_size) + 1] = voxel_size
    # nii.header.set_dim_info()

    return nii


def crop_pad(nii, num_left=None, num_right=None, cval=0):
    """
    Crop or pad the nii around center

    Parameters
    ----------
    nii : ``SpatialImage``
    num_left : list or tuple, optional
    num_right : list or tuple, optional
    cval : float, optional

    Returns
    -------
    out : ``SpatialImage``

    """
    img = nii.get_data()
    shape_old = img.shape
    dims = list(range(img.ndim))
    img_new = sig_util.crop_pad(img, dims, num_left=num_left,
                                num_right=num_right,
                                cval=cval)
    shape_new = img_new.shape

    affine_old = nii.affine
    voxel_size = nii.header._structarr['pixdim'][1:]
    affine_new = _new_affine(shape_old, affine_old, shape_new, voxel_size[:3])

    out = make_nii(img_new, voxel_size=voxel_size, affine=affine_new, header=nii.header)
    return out


def resize(nii, shape_new=None, cval=0):
    """
    Resize the NifTi image with padding or cropping around center.

    Parameters
    ----------
    nii : ``SpatialImage``
        NifTi to resize
    shape_new : list or tuple, optional
        the shape of new image
    cval : float, optional
        The values to set the padded values for all axes.

    Returns
    -------
    out : ``SpatialImage``

    """
    img = nii.get_data()
    shape_old = img.shape
    shape_new = sig_util.to_tuple(shape_new)
    affine_old = nii.affine
    voxel_size = nii.header._structarr['pixdim'][1:]
    affine_new = _new_affine(shape_old, affine_old, shape_new, voxel_size[:3])
    img_new = sig_util.resize(img, shape_new, cval=cval)

    out = make_nii(img_new, voxel_size=voxel_size, affine=affine_new, header=nii.header)
    return out


def zoom(nii, shape_new=None, dtype=None):
    """
    Zoom the nifti.

    Parameters
    ----------
    nii : ``SpatialImage``
        NifTi to zoom
    shape_new : list or tuple, optional
        the shape of new image

    Returns
    -------
    out : ``SpatialImage``

    """
    if shape_new is None:
        return nii

    shape_old = nii.shape
    shape_new = sig_util.to_tuple(shape_new)

    if shape_new == shape_old:
        return nii

    affine_old = nii.affine
    img_new = interp.zoom(nii.get_data(), shape_new, dtype=dtype)
    affine_new = _new_affine(shape_old, affine_old, shape_new)
    out = make_nii(img_new, affine=affine_new, header=nii.header)

    return out


def dcm2niix(read_dir, write_dir=None):
    """
    interface to dcm2niix

    using dcm2niix to converse DICOM TO NIFTI

    Parameters
    ----------
    read_dir : str
        read folder
    write_dir : str, optional
        write folder

    """
    if write_dir and not os.path.exists(write_dir):
        os.makedirs(write_dir, exist_ok=True)
    elif not write_dir:
        write_dir = read_dir

    if platform.system().lower() == 'linux':
        path_exe = '../tools/dcm2niix_linux'
    elif platform.system().lower() == 'windows':
        path_exe = '../tools/dcm2niix_win.exe'
    else:
        raise SystemError('Unknown operating system.')

    cmd = f"{path_exe} -p y -z n -o {write_dir} {read_dir}"

    # subprocess.call(cmd)
    # a = subprocess.run(cmd)
    os.system(cmd)


def dcm2niix_batch(read_dir, write_dir=None):
    """
    batch dcm2niix

    using dcm2niix to converse DICOM TO NIFTI

    Parameters
    ----------
    read_dir : str
        read folder
    write_dir : str, optional
        write folder

    """
    if not write_dir:
        write_dir = read_dir

    subfolders = os.listdir(read_dir)
    if not subfolders:
        raise FileNotFoundError(f'Given read directory "{read_dir}" is empty.')

    dcm_dirs = [os.path.join(read_dir, subfolders[n]) for n in range(len(subfolders))]
    nii_dirs = [os.path.join(write_dir, subfolders[n]) for n in range(len(subfolders))]

    for n in range(len(subfolders)):
        dcm2niix(dcm_dirs[n], nii_dirs[n])


def _new_affine(shape_old, affine_old, shape_new, voxel_size_new=None):
    """
    Compute new affine matrix. Note that the central voxel
    in the old and new image have the same world coordinate.

    Parameters
    ----------
    shape_old : list or tuple
        shape of old image.
        the first three dimensions must be the spatial dimension
    affine_old : ndarray
        4 x 4 affine matrix
    shape_new : list or tuple
        shape of new image.
        the first three dimensions must be the spatial dimension
    voxel_size_new : list or tuple, optional
        voxel_size of the new image. If not provided,
        it will be computed based on the assumption that
        the new and the old image have the same FOV.

    Returns
    -------
    affine_new : ndarray
        4 x 4 affine matrix
    """
    shape_old = sig_util.to_tuple(shape_old)[:3]
    shape_new = sig_util.to_tuple(shape_new)[:3]

    sig_util.check_shape_positive(shape_old)
    sig_util.check_shape_positive(shape_new)

    shape_old = shape_old + (1,) * (3 - len(shape_old))
    shape_new = shape_new + (1,) * (3 - len(shape_new))

    if shape_new == shape_old and voxel_size_new is None:
        return affine_old

    if voxel_size_new is None:
        voxel_size_old = affine_old[[0, 1, 2], [0, 1, 2]]
        voxel_size_new = np.array(shape_old) * voxel_size_old \
                         / np.array(shape_new)

    voxel_size_new = sig_util.to_tuple(voxel_size_new)[:3]
    voxel_size_new = voxel_size_new + (0, ) * (3 - len(voxel_size_new))

    _center = sig_util.arr_center
    _vec = sig_util.vec

    c_old = _vec(_center(shape_old), column=True) + 1
    c_new = _vec(_center(shape_new), column=True) + 1
    c_world_old = np.dot(affine_old[:3, :3], c_old) \
                  + _vec(affine_old[:3, 3], column=True)
    affine_new = np.array(affine_old)
    affine_new[[0, 1, 2], [0, 1, 2]] = voxel_size_new
    affine_new[:3, 3] = _vec(c_world_old - np.dot(affine_new[:3, :3], c_new))

    return affine_new
