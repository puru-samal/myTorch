import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.w_in = A.shape[-1]       # Input Width
        batch_size = A.shape[0]
        in_channels = A.shape[1]
        w_out = self.upsampling_factor * \
            (self.w_in - 1) + 1  # Upsampled output width
        Z = np.zeros((batch_size, in_channels, w_out))  # Initialize Z

        for batch in range(batch_size):
            for channel in range(in_channels):
                i = 0
                for w in range(0, w_out, self.upsampling_factor):
                    Z[batch][channel][w] = A[batch][channel][i]
                    i += 1

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        w_out = dLdZ.shape[-1]
        batch_size = dLdZ.shape[0]
        in_channels = dLdZ.shape[1]
        dLdA = np.zeros((batch_size, in_channels, self.w_in))

        for batch in range(batch_size):
            for channel in range(in_channels):
                i = 0
                for w in range(0, w_out, self.upsampling_factor):
                    dLdA[batch][channel][i] = dLdZ[batch][channel][w]
                    i += 1

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.w_in = A.shape[-1]
        batch_size = A.shape[0]
        in_channels = A.shape[1]
        w_out = int((self.w_in - 1) / self.downsampling_factor + 1)
        Z = np.zeros((batch_size, in_channels, w_out))

        for batch in range(batch_size):
            for channel in range(in_channels):
                i = 0
                for w in range(0, self.w_in, self.downsampling_factor):
                    Z[batch][channel][i] = A[batch][channel][w]
                    i += 1

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size = dLdZ.shape[0]
        in_channels = dLdZ.shape[1]
        dLdA = np.zeros((batch_size, in_channels, self.w_in))  # Initialize Z

        for batch in range(batch_size):
            for channel in range(in_channels):
                i = 0
                for w in range(0, self.w_in, self.downsampling_factor):
                    dLdA[batch][channel][w] = dLdZ[batch][channel][i]
                    i += 1

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        batch_size = A.shape[0]
        channel_size = A.shape[1]
        self.h_in = A.shape[2]
        self.w_in = A.shape[-1]
        h_out = self.upsampling_factor * (self.h_in - 1) + 1
        w_out = self.upsampling_factor * (self.w_in - 1) + 1
        Z = np.zeros((batch_size, channel_size, h_out, w_out))

        for batch in range(batch_size):
            for channel in range(channel_size):
                j = 0
                for h in range(0, h_out, self.upsampling_factor):
                    i = 0
                    for w in range(0, w_out, self.upsampling_factor):
                        Z[batch][channel][h][w] = A[batch][channel][j][i]
                        i += 1
                    j += 1

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size = dLdZ.shape[0]
        channel_size = dLdZ.shape[1]
        h_out = dLdZ.shape[2]
        w_out = dLdZ.shape[-1]
        dLdA = np.zeros((batch_size, channel_size, self.h_in, self.w_in))

        for batch in range(batch_size):
            for channel in range(channel_size):
                j = 0
                for h in range(0, h_out, self.upsampling_factor):
                    i = 0
                    for w in range(0, w_out, self.upsampling_factor):
                        dLdA[batch][channel][j][i] = dLdZ[batch][channel][h][w]
                        i += 1
                    j += 1

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        batch_size = A.shape[0]
        channel_size = A.shape[1]
        self.h_in = A.shape[2]
        self.w_in = A.shape[-1]
        w_out = int((self.w_in - 1) / self.downsampling_factor + 1)
        h_out = int((self.h_in - 1) /
                    self.downsampling_factor + 1)
        Z = np.zeros((batch_size, channel_size, h_out, w_out))

        for batch in range(batch_size):
            for channel in range(channel_size):
                j = 0
                for h in range(0, self.h_in, self.downsampling_factor):
                    i = 0
                    for w in range(0, self.w_in, self.downsampling_factor):
                        Z[batch][channel][j][i] = A[batch][channel][h][w]
                        i += 1
                    j += 1

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size = dLdZ.shape[0]
        channel_size = dLdZ.shape[1]
        dLdA = np.zeros((batch_size, channel_size, self.h_in, self.w_in))

        for batch in range(batch_size):
            for channel in range(channel_size):
                j = 0
                for h in range(0, self.h_in, self.downsampling_factor):
                    i = 0
                    for w in range(0, self.w_in, self.downsampling_factor):
                        dLdA[batch][channel][h][w] = dLdZ[batch][channel][j][i]
                        i += 1
                    j += 1

        return dLdA
