# mripy

A Python based signal processing and MRI reconstruction toolbox. It is built to operate directly on NumPy arrays on CPU.

## Directory

- `data/`: testing data.
- `mripy/mri/`: MRI reconstruction functions, e.g., parallel imaging, compressed sensing.
- `mripy/signal/`: signal processing functions and linear operators.
- `test/`: testing scripts

## Development Plans

### Signal processing functions

- [x] Arbitrary-dimension slice operatiron of numpy array
- [x] convolution and its adjoint
- [x] circular convolution and its adjoint
- [ ] convolution matrix
- [x] operators
- [x] FFT
- [ ] NUFFT
- [x] Spatial transformation
- [x] Spatial transformation of NIFTI
- [x] DICOM TO NIFTI
- [ ] Wavelet
- [ ] Finite difference

### MRI reconstruction functions

#### SENSE

- [x] SENSE
- [ ] CG-SENSE
- [ ] CS-SENSE
- [ ] JSENSE

#### GRAPPA

- [ ] GRAPPA
- [ ] SPIRIT
- [ ] PRUNO

#### Low rank

- [ ] SAKE
- [ ] LORAKS
- [ ] ALOHA
- [ ] ENLIVE

#### Coil sensitivity map and coil combination

- [x] SOS
- [x] Adaptive combination
- [ ] subspace based
- [ ] ESPIRIT

#### Compressive sensing

- [ ] CS (soft-threshold)
- [ ] CS (CG, L1-norm)

#### NIFTI

- [ ] NIFTI class (interface to AntsImage and ScalarImage)

## Requirements

- Numpy
- Nibabel
