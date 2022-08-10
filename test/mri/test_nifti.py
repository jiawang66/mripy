# -*- coding: utf-8 -*-
"""
Test NIFTI
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.append('../../../mripy')
from mripy.mri import nifti


def main():
    xshape = (10, 11, )
    voxel_size = (2, 3, 2.5, 5)
    # affine = np.eye(4)
    # affine[:3, -1] = [10, 20, -10]
    affine = None
    img = np.array(range(np.prod(xshape))).reshape(xshape)
    nii_1 = nifti.make_nii(img, voxel_size=voxel_size, affine=affine)

    # nii_1 = nifti.load_nii('../../data/mgre.nii')
    # xshape = nii_1.shape

    path_save = '../../results/'
    if not os.path.isdir(path_save):
        os.makedirs(path_save)

    yshape = tuple([round(j * 1.5) if i < 3 else j for i,j in enumerate(xshape)])
    nii_2 = nifti.resize(nii_1, shape_new=yshape, cval=0)

    nifti.save_nii(nii_1, path_save + 'nii_1.nii')
    nifti.save_nii(nii_2, path_save + 'nii_2.nii')


if __name__ == '__main__':
    main()
