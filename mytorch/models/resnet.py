import os
import numpy as np
from mytorch.nn.batchnorm import BatchNorm2d
from mytorch.nn.activation import *
from mytorch.nn.Conv2d import *
import sys


class Conv_BN(object):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # TODO
        self.layers = [Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                       BatchNorm2d(out_channels)]

    def forward(self, A):
        # TODO
        for layer in self.layers:
            A = layer.forward(A)

        return A

    def backward(self, grad):
        # TODO
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad


class ResBlock(object):

    def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):

        self.ConvBlock = [Conv_BN(in_channels, out_channels, filter_size, stride, padding),
                          ReLU(),
                          Conv_BN(out_channels, out_channels, 1, 1, 0)]  # TODO Initialize the ConvBlock layers in this list.

        self.final_activation = ReLU()  # TODO

        if stride != 1 or in_channels != out_channels or filter_size != 1 or padding != 0:
            self.residual_connection = Conv_BN(in_channels, out_channels,
                                               filter_size, stride, padding)  # TODO
        else:
            self.residual_connection = Identity()  # TODO

    def forward(self, A):

        Z = A
        '''
            Implement the forward for convolution layer.

            '''
        # TODO
        for layer in self.ConvBlock:
            A = layer.forward(A)

            '''
                Add the residual connection to the output of the convolution layers

                '''
            # TODO
            out = self.residual_connection.forward(Z) + A

            '''
		Pass the the sum of the residual layer and convolution layer to the final activation function
		'''
            # TODO
        out = self.final_activation.forward(out)

        return out

    def backward(self, grad):
        '''
        Implement the backward of the final activation
        '''
        # TODO
        grad = self.final_activation.backward(grad)

        '''
            Implement the backward of residual layer to get "residual_grad"
            '''
        # TODO
        residual_grad = self.residual_connection.backward(grad)

        '''
            Implement the backward of the convolution layer to get "convlayers_grad"
            '''
        # TODO
        for layer in reversed(self.ConvBlock):
            grad = layer.backward(grad)

            '''
                Add convlayers_grad and residual_grad to get the final gradient 
                '''
            # TODO
        grad += residual_grad
        return grad
