import numpy as np
from ann.activations import Activation


class NeuralLayer:

    def __init__(self, input_dim, output_dim, activation=None, weight_init="xavier"):

        self.activation = Activation(activation) if activation else None

        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

    def forward(self, X):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.A_prev = X
        self.Z = X @ self.W + self.b

        if self.activation:
            return self.activation.forward(self.Z)

        return self.Z

    def backward(self, grad_output):

        if self.activation:
            grad_output = grad_output * self.activation.backward(self.Z)

        self.grad_W = self.A_prev.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        return grad_output @ self.W.T
