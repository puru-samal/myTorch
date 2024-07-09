import numpy as np

class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0] 
        self.C = A.shape[1]  
        se = (self.A - self.Y) * (self.A - self.Y) 
        sse = np.ones((1, self.N)) @ se @ np.ones((self.C, 1))
        mse = sse / (self.N * self.C)
        return mse

    def backward(self):
        dLdA = 2.0 * (self.A - self.Y) / (self.N * self.C)
        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0] 
        C = A.shape[1] 

        Ones_C = np.ones((C, 1))  
        Ones_N = np.ones((self.N, 1)) 

        expA = np.exp(A)
        sums = np.sum(expA, axis=1)                # sums.shape == (N, )
        sums = np.expand_dims(sums, axis=1)        # sums.shape == (N, 1)
        inv_sums = np.reciprocal(sums)
        self.softmax = expA * inv_sums 

        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C  
        sum_crossentropy = np.transpose(Ones_N) @ crossentropy 
        L = sum_crossentropy / self.N
        return L

    def backward(self):
        dLdA = (self.softmax - self.Y) / self.N  # TODO
        return dLdA
