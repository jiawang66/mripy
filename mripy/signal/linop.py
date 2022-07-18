# -*- coding: utf-8 -*-
"""Linear operators

This module contains an abstraction class Linop for linear operators,
and provides commonly used linear operators, including signal transforms
such as FFT, NUFFT, and wavelet, and array manipulation operators,
such as reshape, transpose, resize, slice and embed.

"""


import numpy as np
from . import backend, util


class Linop():
    """Abstraction for linear operator.

    Linop can be called on a NumPy to perform a linear operation.

    Given a Linop A, and an appropriately shaped input x, the following are
    both valid operations to compute x -> A(x):

       >> y = A * x
       >> y = A(x)

    Its adjoint linear operator can be obtained using the .H attribute.
    Linops can be scaled, added, subtracted, stacked and composed.
    Here are some example of valid operations on Linop A, Linop B,
    and a scalar a:

       >> A.H
       >> a * A + B
       >> a * A * B

    Args:
        oshape: Output shape.
        ishape: Input shape.
        repr_str (string or None): default: class name.

    Attributes:
        oshape: output shape.
        ishape: input shape.
        H: adjoint linear operator.
        N: normal linear operator.

    """

    def __init__(self, oshape, ishape, repr_str=None):
        util.check_shape_positive(oshape)
        util.check_shape_positive(ishape)

        self.oshape = oshape
        self.ishape = ishape

        if repr_str is None:
            self.repr_str = self.__class__.__name__
        else:
            self.repr_str = repr_str

        self.adj = None
        self.normal = None

    def _check_ishape(self, input):
        if not all([i1==i2 for i1, i2 in zip(self.ishape, input.shape)]):
            raise ValueError(f'Input shape mismatch for{self}, got {input.shape}')

    def _check_oshape(self, output):
        if not all([i1 == i2 for i1, i2 in zip(self.ishape, output.shape)]):
            raise ValueError(f'Output shape mismatch for{self}, got {output.shape}')

    def _apply(self, input):
        raise NotImplementedError

    def apply(self, input):
        """Apply linear operation on input.

        This function checks for the input/output shapes and devices,
        and calls the internal user-defined _apply() method.

        Args:
            input (ndarray): input array of shape `ishape`.

        Returns:
            ndarray: output array of shape `oshape`.

        """
        try:
            self._check_ishape(input)
            output = self._apply(input)
            self._check_oshape(output)
        except Exception as e:
            raise RuntimeError(f'Exceptions from {self}.') from e

        return output

    def _adjoint_linop(self):
        raise NotImplementedError

    def _normal_linop(self):
        return self.H * self

    @property
    def H(self):
        r"""Return adjoint linear operator.

        An adjoint linear operator :math:`A^H` for
        a linear operator :math:`A` is defined as:

        .. math:
            \left< A x, y \right> = \left< x, A^H, y \right>

        Returns:
            Linop: adjoint linear operator.

        """
        if self.adj is None:
            self.adj = self._adjoint_linop()
        return self.adj

    @property
    def N(self):
        r"""Return normal linear operator.

        A normal linear operator :math:`A^HA` for
        a linear operator :math:`A`.

        Returns:
            Linop: normal linear operator.

        """
        if self.normal is None:
            self.normal = self._normal_linop()
        return self.normal

    def __call__(self, input):
        return self.__mul__(input)

    def __mul__(self, input):
        if isinstance(input, Linop):
            return Compose([self, input])
        elif np.isscalar(input):
            M = Multiply(self.ishape, input)
            return Compose([self, M])
        elif isinstance(input, backend.get_array_module(input).ndarray):
            return self.apply(input)

        return NotImplemented

    def __rmul__(self, input):
        if np.isscalar(input):
            M = Multiply(self.oshape, input)
            return Compose([M, self])

        return NotImplemented

    def __add__(self, input):
        if isinstance(input, Linop):
            return Add([self, input])

        raise NotImplementedError

    def __neg__(self):
        return -1 * self

    def __sub__(self, input):
        return self.__add__(-input)

    def __repr__(self):
        return f'<{self.oshape}x{self.ishape}> {self.repr_str} Linop>'


class Add(Linop):
    """Addition of linear operators.

    Input and output shapes and devices must match.

    Args:
        linops (list of Linops): Input linear operators.

    Returns:
        Linop: linops[0] + linops[1] + ... + linops[n - 1]

    """

    def __init__(self, linops):
        _check_linops_same_ishape(linops)
        _check_linops_same_oshape(linops)

        self.linops = linops
        oshape = linops[0].oshape
        ishape = linops[0].ishape

        super().__init__(oshape, ishape, repr_str=' + '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):
        output = 0
        with backend.get_device(input):
            for linop in self.linops:
                output += linop(input)

        return output

    def _adjoint_linop(self):
        return Add([linop.H for linop in self.linops])


class Compose(Linop):
    """Composition of linear operators.

    Args:
        linops (list of Linops): Linear operators to be composed.

    Returns:
        Linop: linops[0] * linops[1] * ... * linops[n - 1]

    """

    def __init__(self, linops):
        _check_compose_linops(linops)
        self.linops = _combine_compose_linops(linops)

        super().__init__(self.linops[0].oshape,
                         self.linops[-1].ishape,
                         repr_str=' * '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):
        output = input
        for linop in self.linops[::-1]:
            output = linop(output)

        return output

    def _adjoint_linop(self):
        return Compose([linop.H for linop in self.linops[::-1]])


class Identity(Linop):
    """Identity linear operator.

    Return input directly.

    Args:
        shape (list or tuple of ints): Input shape

    """

    def __init__(self, shape):
        super().__init__(shape, shape)

    def _apply(self, input):
        return input

    def _adjoint_linop(self):
        return self

    def _normal_linop(self):
        return self


class Multiply(Linop):
    """Multiplication linear operator.

    Hadamard multiplication or array (support broadcasting)

    Args:
        ishape (list or tuple of ints): Input shape.
        mult (ndarray or scalar): Array or scalar to multiply.

    """

    def __init__(self, ishape, mult, conj=False):
        self.mult = mult
        self.conj = conj
        if np.isscalar(mult):
            self.mshape = [1]
        else:
            self.mshape = mult.shape

        oshape = _get_multiply_oshape(ishape, self.mshape)
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        xp = device.xp
        with device:
            if np.isscalar(self.mult):
                if self.mult == 1:
                    return input

                mult = self.mult
                if self.conj:
                    mult = mult.conjugate()

            else:
                mult = backend.to_device(self.mult, backend.get_device(input))
                if self.conj:
                    mult = xp.conj(mult)

            return input * mult

    def _adjoint_linop(self):
        '''
        Since that Multiply supports broadcasting, so for its adjoint
        we need to sum over the broadcasting axes
        '''
        sum_axes = _get_multiply_adjoint_sum_axes(self.oshape, self.ishape, self.mshape)

        M = Multiply(self.oshape, self.mult, conj=not self.conj)
        S = Sum(M.oshape, sum_axes)
        R = Reshape(self.ishape, S.oshape)
        return R * S * M


class Reshape(Linop):
    """Reshape input to given output shape.

    Args:
        oshape (list or tuple of ints): Output shape.
        ishape (list or tuple of ints): Input shape.

    """

    def __init__(self, oshape, ishape):
        super().__init__(oshape, ishape)

    def _apply(self, input):
        with backend.get_device(input):
            return input.reshape(self.oshape)

    def _adjoint_linop(self):
        return Reshape(self.ishape, self.oshape)

    def _normal_linop(self):
        return Identity(self.ishape)


class Sum(Linop):
    """Sum linear operator. Accumulate axes by summing.

    Args:
        ishape (list or tuple of ints): Input shape.
        axes (list or tuple of ints): Axes to sum over.
    """

    def __init__(self, ishape, axes):
        self.axes = tuple(i % len(ishape) for i in axes)
        oshape = [ishape[i] for i in range(len(ishape)) if i not in self.axes]

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        xp = device.xp
        with device:
            return xp.sum(input, axis=self.axes)

    def _adjoint_linop(self):
        return Tile(self.ishape, self.axes)


class Tile(Linop):
    """Tile linear operator.

    Args:
        oshape (list or tuple of ints): Output shape.
        axes (list or tuple of ints): Axes to tile.

    """

    def __init__(self, oshape, axes):
        self.axes = tuple(i % len(oshape) for i in axes)
        ishape = [oshape[d] for d in range(len(oshape)) if d not in self.axes]
        self.expanded_ishape = []
        self.reps = []
        for d in range(len(oshape)):
            if d in self.axes:
                self.expanded_ishape.append(1)
                self.reps.append(oshape[d])
            else:
                self.expanded_ishape.append(oshape[d])
                self.reps.append(1)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        xp = device.xp
        with device:
            return xp.tile(input.reshape(self.expanded_ishape), self.reps)

    def _adjoint_linop(self):
        return Sum(self.oshape, self.axes)


def _check_compose_linops(linops):
    for linop1, linop2 in zip(linops[:-1], linops[1:]):
        if (linop1.ishape != linop2.oshape):
            raise ValueError(f'cannot compose {linop1} and {linop2}.')


def _check_linops_same_ishape(linops):
    for linop in linops:
        if (linop.ishape != linops[0].ishape):
            raise ValueError(f'Linops must have the same ishape, got {linops}.')


def _check_linops_same_oshape(linops):
    for linop in linops:
        if (linop.oshape != linops[0].oshape):
            raise ValueError(f'Linops must have the same oshape, got {linops}.')


def _combine_compose_linops(linops):
    combined_linops = []
    for linop in linops:
        if isinstance(linop, Compose):
            combined_linops += linop.linops
        else:
            combined_linops.append(linop)

    return combined_linops


def _get_multiply_adjoint_sum_axes(oshape, ishape, mshape):
    ishape_exp, mshape_exp = util.expand_shapes(ishape, mshape)
    max_ndim = max(len(ishape), len(mshape))
    sum_axes = []

    for i, m, o, d in zip(ishape_exp, mshape_exp, oshape, range(max_ndim)):
        if (i == 1 and (m != 1 or o != 1)):
            sum_axes.append(d)

    return sum_axes


def _get_multiply_oshape(ishape, mshape):
    ishape_exp, mshape_exp = util.expand_shapes(ishape, mshape)
    max_ndim = max(len(ishape), len(mshape))
    oshape = []

    '''
    Comments: the array can be broadcat if size of one dimension is 1.
    '''
    for i, m, d in zip(ishape_exp, mshape_exp, range(max_ndim)):
        if not (i == m or i == 1 or m == 1):
            raise ValueError(f'Invalid shapes: {ishape}, {mshape}.')

        oshape.append(max(i, m))

    return oshape
