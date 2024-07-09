import numpy as np
from .resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, h_in, w_in = A.shape[0], A.shape[2], A.shape[3]
        w_out = w_in - self.kernel_size + 1
        h_out = h_in - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, h_out, w_out))

        # A : N * C_in * h_in * w_in
        # Z : N * C_out * h_out * w_out
        # W : C_out * C_in * K * K

        for h in range(h_out):
            for w in range(w_out):
                axs = ([1, 2, 3], [1, 2, 3])
                Z[:, :, h, w] += np.tensordot(
                    A[:, :, h:h+self.kernel_size, w:w+self.kernel_size], self.W, axes=axs)
                Z[:, :, h, w] += self.b  # Should automatically broadcast

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, h_out, w_out = dLdZ.shape[0], dLdZ.shape[2], dLdZ.shape[3]
        h_in, w_in = self.A.shape[2], self.A.shape[3]

        # dLdb : C_out x 1

        self.dLdb += np.sum(dLdZ, axis=(0, 2, 3))

        # dLdW : C_out * C_in * K * K

        for kh in range(self.kernel_size):
            for kw in range(self.kernel_size):
                axs = ([0, 2, 3], [0, 2, 3])
                self.dLdW[:, :, kh, kw] += np.tensordot(
                    dLdZ, self.A[:, :, kh:kh+h_out, kw:kw + w_out], axes=axs)

        # dLdA : N * C_in * h_in * w_in

        dLdA = np.zeros((batch_size, self.in_channels, h_in, w_in))
        # Padding only on laxt two axis
        pwidths = ((0,), (0,), (self.kernel_size-1,), (self.kernel_size-1,))
        pdLdZ = np.pad(dLdZ, pad_width=pwidths, mode='constant')
        flipW = np.flip(self.W, axis=(2, 3))  # Flip only on last two axis

        for h in range(h_in):
            for w in range(w_in):
                axs = ([1, 2, 3], [0, 2, 3])
                dLdA[:, :, h, w] = np.tensordot(
                    pdLdZ[:, :, h:h+self.kernel_size, w:w+self.kernel_size], flipW, axes=axs)

        self.dLdA = dLdA
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        # TODO
        A = np.pad(A, pad_width=((0,), (0,), (self.pad,),
                   (self.pad,)), mode='constant')

        # Call Conv2d_stride1
        # TODO
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        # TODO
        dLdA = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA)  # TODO

        # Unpad the gradient
        # TODO
        dLdA = dLdA[:, :, self.pad:dLdA.shape[-1] -
                    self.pad, self.pad:dLdA.shape[-1]-self.pad]

        return dLdA
