# Convolution

Convolution and its adjoint

## Concepts

Given data $x$ and fileter $d$, then the convolution of $x$ and $d$ is:

$$
y[n] = conv(x, d) = \sum_i x[n] \cdot d[i-n]
$$

## Centers

Let the shape of an array is $s$, we set the center of array as

$$
c = ceil(s / 2)
$$

That says, if s is an even number, then $c$ is on the left of $s/2$

Note that, the center definitions of `scipy.signal.convolve` and `scipy.ndimage.convolve` have a little different

For `scipy.signal.convolve` : $c = ceil(s / 2)$
For `scipy.ndimage.convolve` : $c = ceil((s + 1) / 2)$

In this project, we uniformly set that the center of the array is $c = ceil(s / 2)$.

So the subscript of center is $c = ceil(s / 2) - 1$

## Adjoint with respect to data

## Adjoint with respect to filter
