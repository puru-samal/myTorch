import numpy as np

class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = self.Z.shape[0]  # Batch Size
        self.M = np.sum(self.Z, axis=0) / self.N  # TODO
        self.V = np.sum((self.Z - self.M)**2, axis=0) / self.N  # TODO

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / \
                np.sqrt(self.V + self.eps)  # Normalized Z

            self.running_M = self.alpha * self.running_M + \
                (1 - self.alpha) * self.M  # Running Mean

            self.running_V = self.alpha * self.running_V + \
                (1 - self.alpha) * self.V  # Running Variance

            self.BZ = self.NZ * self.BW + self.Bb  # Output of BN Layer
        else:
            # inference mode
            NZ = (self.Z - self.running_M) / \
                np.sqrt(self.running_V + self.eps)   # Normalized Z

            self.BZ = NZ * self.BW + self.Bb  # Output of BN Layer

        return self.BZ

    def backward(self, dLdBZ):

        # Change in loss wrt change in weight scaling params
        self.dLdBW = np.sum(dLdBZ, axis=0)
        # # Change in loss wrt change in bias shifting params
        self.dLdBb = np.sum(dLdBZ * self.NZ, axis=0)

        dLdNZ = dLdBZ * self.BW  # Derivative of loss wrt norm Z

        sqrtVar = np.sqrt(self.V + self.eps)
        zu = self.Z - self.M

        dLdV = -0.5 * np.sum((dLdNZ * (zu / (sqrtVar**3))), axis=0)   # TODO
        dLdM = -np.sum(dLdNZ / sqrtVar, axis=0) - dLdV * \
            (2 / self.N) * np.sum(zu, axis=0)  # TODO

        dLdZ = (dLdNZ / sqrtVar) + (dLdV * (2 / self.N) * zu) + \
            (dLdM / self.N)  # TODO

        return dLdZ
    
class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        self.Z = Z
        self.N = self.Z.shape[0] * self.Z.shape[2] * \
            self.Z.shape[3]  # Batch Size
        self.M = np.sum(self.Z, axis=(0, 2, 3),
                        keepdims=True) / self.N  # TODO
        self.V = np.sum((self.Z - self.M)**2, axis=(0, 2, 3),
                        keepdims=True) / self.N  # TODO

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / \
                np.sqrt(self.V + self.eps)  # Normalized Z

            self.running_M = self.alpha * self.running_M + \
                (1 - self.alpha) * self.M  # Running Mean

            self.running_V = self.alpha * self.running_V + \
                (1 - self.alpha) * self.V  # Running Variance

            self.BZ = self.NZ * self.BW + self.Bb  # Output of BN Layer
        else:
            # inference mode
            NZ = (self.Z - self.running_M) / \
                np.sqrt(self.running_V + self.eps)   # Normalized Z

            self.BZ = NZ * self.BW + self.Bb  # Output of BN Layer

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=(0, 2, 3),
                            keepdims=True)  # TODO
        self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3),
                            keepdims=True)  # TODO

        dLdNZ = dLdBZ * self.BW  # Derivative of loss wrt norm Z
        sqrtVar = np.sqrt(self.V + self.eps)
        zu = self.Z - self.M

        dLdV = -0.5 * np.sum((dLdNZ * (zu / (sqrtVar**3))),
                             axis=(0, 2, 3), keepdims=True)   # TODO
        dLdM = -np.sum(dLdNZ / sqrtVar, axis=(0, 2, 3), keepdims=True) - dLdV * \
            (2 / self.N) * np.sum(zu, axis=(0, 2, 3), keepdims=True)  # TODO

        dLdZ = (dLdNZ / sqrtVar) + (dLdV * (2 / self.N) * zu) + \
            (dLdM / self.N)  # TODO

        return dLdZ

