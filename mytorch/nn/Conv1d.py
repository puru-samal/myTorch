# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from .resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size = A.shape[0]
        w_out = A.shape[-1] - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, w_out))

        # A : N * C_in * w_in
        # Z : N * C_out * w_out
        # W : C_out * C_in * K

        for w in range(w_out):
            axs = ([1, 2], [1, 2])
            Z[:, :, w] += np.tensordot(self.A[:, :,
                                       w:w+self.kernel_size], self.W, axes=axs)
            Z[:, :, w] += self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        batch_size = dLdZ.shape[0]
        w_out = dLdZ.shape[-1]
        w_in = self.A.shape[-1]

        # dLdb : C_out x 1

        self.dLdb += np.sum(dLdZ, axis=(0, 2))

        # dLdW : C_out * C_in * K

        for k in range(self.kernel_size):
            axs = ([0, 2], [0, 2])
            self.dLdW[:, :,
                      k] += np.tensordot(dLdZ, self.A[:, :, k:k+w_out], axes=axs)

        # dLdA  : N x C_in * w_in
        # pdLdZ : N * C_out * w_in
        # W     : C_out * C_in * K

        dLdA = np.zeros((batch_size, self.in_channels, w_in))
        # Padding only on laxt axis
        pwidths = ((0,), (0,), (self.kernel_size-1,))
        pdLdZ = np.pad(dLdZ, pad_width=pwidths, mode='constant')
        flipW = np.flip(self.W, axis=2)  # Flip only on last axis

        for w in range(w_in):
            axs = ([1, 2], [0, 2])
            dLdA[:, :, w] = np.tensordot(
                pdLdZ[:, :, w:w+self.kernel_size], flipW, axes=axs)

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        # TODO
        if self.pad > 0:
            A = np.pad(A, pad_width=((0,), (0,), (self.pad,)), mode='constant')

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdA = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        # TODO
        dLdA = self.conv1d_stride1.backward(dLdA)

        # Unpad the gradient
        # TODO
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:(dLdA.shape[-1]-self.pad)]

        return dLdA
