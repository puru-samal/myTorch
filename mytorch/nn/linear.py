import numpy as np

class Linear:
    
    def __init__(self, in_features, out_features):
        self.W = np.zeros((out_features, in_features))  # W.shape = (C1, C0)
        self.b = np.zeros((out_features, 1))  # b.shape = (C1, 1)
        self.A = None  # Inputs
        self.N = None  # Batch Size
        self.dLdW = None  # Change in loss wrt change in weights
        self.dLdb = None  # Change in loss wrt change in biases

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        """
        self.A = A  # A.shape == (N, C0)
        self.N = self.A.shape[0]  # Store the batch size of input
        Z = self.A @ self.W.T + self.b.T
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient wrt layer output (N, C1)
        :return: Gradient wrt layer input (N, C0)
        """
        self.dLdA = dLdZ @ self.W  # dLdA.shape == (N, Cin)
        self.dLdW = dLdZ.T @ self.A  # dLdW.shape == (Cout, Cin)
        self.dLdb = dLdZ.T @ np.ones((self.N, 1))  # dLdb.shape = (Cout, 1)
        return self.dLdA
