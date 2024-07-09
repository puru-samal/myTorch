import numpy as np

class Dropout1d(object):
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

class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p
        self.mask = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.
        # TODO

        if eval:
            return x
        else:

            ch_selector = np.random.binomial(
                1, 1 - self.p, size=(x.shape[0], x.shape[1], 1, 1))
            self.mask = np.tile(ch_selector, (1, 1, x.shape[2], x.shape[3]))
            return (x * self.mask) / (1-self.p)

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule
        # TODO
        return delta * self.mask / (1-self.p)

