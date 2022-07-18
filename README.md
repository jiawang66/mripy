# mripy

A Python based signal processing and MRI reconstruction toolbox. It is built to operate directly on NumPy arrays on CPU.

## Directory

- `data/`: testing data.
- `mripy/mri/`: MRI reconstruction functions, e.g., parallel imaging, compressed sensing and machine-learning based reconstruction.
- `mripy/signal/`: signal processing functions and linear operators.
- `test/`: testing scripts

## Plans

`mripy/mri/`

- [ ] SENSE
- [ ] GRAPPA
- [ ] coil combination with phase
- [ ] CS (soft-threshold)
- [ ] CS (CG, L1-norm)
- [ ] Sensitivity map computation (fitting)
- [ ] GSENSE
- [ ] JSENSE
- [ ] SPIRIT
- [ ] SAKE
- [ ] LORAKS
- [ ] ESPIRIT
- [ ] ALOHA
- [ ] ENLIVE

`mripy/signal/`

- [ ] Arbitrary-dimension slice operatiron of numpy array
- [ ] convolution
- [ ] convolution matrix
- [ ] operators
- [ ] FFT
- [ ] NUFFT
- [ ] Spatial transformation
- [ ] Spatial transformation of NIFTI
- [ ] DICOM TO NIFTI
- [ ] Wavelet
- [ ] Finite difference
- [ ] NIFTI class (interface to AntsImage and ScalarImage)
- [ ] $\cdots$

`test/`

## Requirements

- Numpy
- Nibabel
