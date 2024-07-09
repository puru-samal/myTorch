import numpy as np
from .resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.batch_size, self.in_channels, self.w_in, self.h_in = A.shape

        out_channels, w_out, h_out = self.in_channels, self.w_in - \
            self.kernel + 1, self.h_in - self.kernel + 1

        self.maxindex = np.empty(
            (self.batch_size, out_channels, w_out, h_out), dtype=tuple)

        Z = np.zeros((self.batch_size, out_channels, w_out, h_out))

        for batch in range(self.batch_size):
            for ch in range(out_channels):
                for w in range(w_out):
                    for h in range(h_out):
                        scan = A[batch][ch][w:w+self.kernel, h:h+self.kernel]
                        Z[batch][ch][w][h] = np.max(scan)
                        self.maxindex[batch][ch][w][h] = np.unravel_index(
                            np.argmax(scan), scan.shape)
                        self.maxindex[batch][ch][w][h] = tuple(
                            np.add((w, h), self.maxindex[batch][ch][w][h]))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(
            (self.batch_size, self.in_channels, self.w_in, self.h_in))
        out_channels, w_out, h_out = dLdZ.shape[1], dLdZ.shape[2], dLdZ.shape[3]

        for batch in range(self.batch_size):
            for ch in range(out_channels):
                for w in range(w_out):
                    for h in range(h_out):
                        i1, i2 = self.maxindex[batch][ch][w][h]
                        dLdA[batch][ch][i1, i2] += dLdZ[batch][ch][w][h]
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.batch_size, self.in_channels, self.w_in, self.h_in = A.shape
        w_out, h_out = self.w_in - self.kernel + 1, self.h_in - self.kernel + 1
        Z = np.zeros((self.batch_size, self.in_channels, w_out, h_out))

        for w in range(w_out):
            for h in range(h_out):
                Z[:, :, w, h] = np.mean(
                    A[:, :, w:w+self.kernel, h:h+self.kernel], axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(
            (self.batch_size, self.in_channels, self.w_in, self.h_in))

        pwidths = ((0,), (0,), (self.kernel-1,), (self.kernel-1,))
        # Pad with zeroes to shape match
        pdLdZ = np.pad(dLdZ, pad_width=pwidths, mode='constant')

        for w in range(self.w_in):
            for h in range(self.h_in):
                dLdA[:, :, w, h] = np.mean(pdLdZ[:, :, w:w +
                                                 self.kernel, h:h+self.kernel], axis=(2, 3))
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)

        return dLdA
