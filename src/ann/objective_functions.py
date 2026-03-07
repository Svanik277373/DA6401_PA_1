"""
Loss/Objective Functions and Their Derivatives
"""

import numpy as np


class Loss:

    def __init__(self, name="cross_entropy"):
        self.name = name

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, y_true, logits):

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

        if self.name == "cross_entropy":

            return (self.probs - self.y_true) / self.y_true.shape[0]

        elif self.name == "mse":

            return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]