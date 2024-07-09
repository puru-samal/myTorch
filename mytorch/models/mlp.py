import numpy as np
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """
        self.layers = [Linear(2, 3), ReLU()]  # 2 out_features, 3 in_features
        self.dLdZ0 = None  # Change in loss wrt change in Z0: dLdA * dAdZ
        self.dLdA0 = None  # Chanve in loss wrt change in A0: dLdZ0 * dZ0dA0

    def forward(self, A0):
        """
        Pass the input through the linear layer followed by the activation layer to get the model output.
        """
        # Output of linear layer: Z0 = W * A0 + b
        self.Z0 = self.layers[0].forward(A0)  # Z0.shape == (N, Cout)

        # Output activation: A1 = f(Z0)
        A1 = self.layers[1].forward(self.Z0)  # A1.shape == (N, Cout)
        return A1

    def backward(self, dLdA1):
        """
        Backpropogation through the model.
        """
        self.dLdZ0 = self.layers[1].backward(dLdA1)       # dLdZ0.shape == (N, Cout)
        self.dLdA0 = self.layers[0].backward(self.dLdZ0)  # dLdA0.shape == (N, Cin)
        return self.dLdA0


class MLP1:

    def __init__(self):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """
        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]

    def forward(self, A0):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        self.Z0 = self.layers[0].forward(A0)  
        self.A1 = self.layers[1].forward(self.Z0)  
        self.Z1 = self.layers[2].forward(self.A1) 
        self.A2 = self.layers[3].forward(self.Z1)
        return self.A2

    def backward(self, dLdA2):
        """
        Backpropogation through the model.
        """

        self.dLdZ1 = self.layers[3].backward(dLdA2)  
        self.dLdA1 = self.layers[2].backward(self.dLdZ1) 
        self.dLdZ0 = self.layers[1].backward(self.dLdA1) 
        self.dLdA0 = self.layers[0].backward(self.dLdZ0)
        return self.dLdA0


class MLP4:
    def __init__(self):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagramatic view in the writeup for better understanding.
        Use ReLU activation function for all the linear layers.)
        """

        # List of Hidden and activation Layers in the correct order
        self.layers = [Linear(2, 4), ReLU(), Linear(4, 8), ReLU(), 
                       Linear(8, 8), ReLU(), Linear(8, 4), ReLU(), 
                       Linear(4, 2), ReLU()] 


    def forward(self, A):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """
        self.A = [A] # For test to verify correctness
        for layer in self.layers:
            A = layer.forward(A) 
            self.A.append(A) 

        return A

    def backward(self, dLdA):
        """
        Backpropogation through the model.
        """
        self.dLdA = [dLdA] # For test to verify correctness
        for layer in reversed(self.layers):
            dLdA = layer.backward(dLdA) 
            self.dLdA = [dLdA] + self.dLdA

        return dLdA
