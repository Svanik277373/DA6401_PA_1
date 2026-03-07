import numpy as np


class Activation:

    def __init__(self, name):

        self.name = name

        if name not in ["relu", "sigmoid", "tanh"]:
            raise ValueError("Unsupported activation")

    def forward(self, x):

        self.x = x

        if self.name == "relu":
            return np.maximum(0, x)

        if self.name == "sigmoid":
            return 1 / (1 + np.exp(-x))

        if self.name == "tanh":
            return np.tanh(x)

    def backward(self, grad_output):

        if self.name == "relu":
            grad = (self.x > 0).astype(float)

        elif self.name == "sigmoid":
            s = 1 / (1 + np.exp(-self.x))
            grad = s * (1 - s)

        elif self.name == "tanh":
            grad = 1 - np.tanh(self.x) ** 2

        return grad_output * grad