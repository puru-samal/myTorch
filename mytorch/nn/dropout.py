import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # TODO: Generate mask and apply to x
            self.mask = np.random.binomial(1, 1 - self.p, x.shape)
            x *= self.mask
            x /= 1 - self.p
            return x
        else:
            return x

    def backward(self, delta):
        # TODO: Multiply mask with delta and return
        return delta * self.mask
