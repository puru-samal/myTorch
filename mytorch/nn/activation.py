import numpy as np
import scipy
import math


class Identity:

    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:

    def forward(self, Z):
        self.A =  1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):

        dAdZ = self.A - self.A * self.A
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = 1.0 - self.A**2 
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:

    def forward(self, Z):
        self.A = np.maximum(0, Z) 
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0.0, 1.0, 0.0)
        dLdZ = dLdA * dAdZ 
        return dLdZ


class GELU:

    def forward(self, Z):
        self.term1 = 0.5 * (1.0 + scipy.special.erf(Z / np.sqrt(2.0)))
        self.A = Z * self.term1 
        self.Z = Z
        return self.A

    def backward(self, dLdA):
        dAdZ = self.term1 + (self.Z / np.sqrt(2 * np.pi)) * np.exp((self.Z**2) * -0.5)
        dLdZ = dLdA * dAdZ 
        return dLdZ


class Softmax:

    def forward(self, Z):
        expZ = np.exp(Z)
        sums = np.sum(expZ, axis=1)                # sums.shape == (N, )
        sums = np.expand_dims(sums, axis=1)        # sums.shape == (N, 1)
        inv_sums = np.reciprocal(sums)
        self.A = expZ * inv_sums 
        return self.A

    def backward(self, dLdA):
        # Calculate the batch size and number of features
        N = dLdA.shape[0]  
        C = dLdA.shape[1]  

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C)) 

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))  

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    J[m, n] = self.A[i, m] * \
                        (1.0 - self.A[i, m]) if m == n else - self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i, :] = dLdA[i, :] @ J 

        return dLdZ
