# -*- coding: utf-8 -*-
"""
Test NIFTI
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import mripy.mri as mri
import mripy.signal as sig


def main():
    xshape = (10, ) * 2
    voxel_size = (2, 3, 4)
    # affine = np.eye(4)
    # affine[:3, -1] = [10, 20, -10]
    affine = None
    img = np.array(range(np.prod(xshape))).reshape(xshape)
    nii = mri.nifti.make_nii(img, voxel_size=voxel_size, affine=affine)

    nii.header.get_best_affine()


if __name__ == '__main__':
    main()
