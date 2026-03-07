import numpy as np


class Loss:

    def __init__(self, name="cross_entropy"):
        self.name = name

    def softmax(self, z):

        if z.ndim == 1:
            z = z.reshape(1, -1)

        z = z - np.max(z, axis=1, keepdims=True)

        exp = np.exp(z)

        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, y_true, logits):

        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        if self.name == "cross_entropy":

            self.probs = self.softmax(logits)
            self.y_true = y_true

            m = y_true.shape[0]

            loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / m

            return loss

        elif self.name == "mse":

            self.y_true = y_true
            self.y_pred = logits

            return np.mean((y_true - logits) ** 2)

    def backward(self):

        m = self.y_true.shape[0]

        if self.name == "cross_entropy":

            grad = self.probs - self.y_true
            grad = grad / m

            return grad

        elif self.name == "mse":

            return 2 * (self.y_pred - self.y_true) / m
