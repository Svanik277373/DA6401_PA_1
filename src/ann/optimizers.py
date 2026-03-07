"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np


class SGD:

    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, layers):

        for layer in layers:

            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class Momentum:

    def __init__(self, lr=0.001, beta=0.9):

        self.lr = lr
        self.beta = beta

        self.vW = []
        self.vb = []

        self.initialized = False

    def step(self, layers):

        if not self.initialized:

            for layer in layers:

                self.vW.append(np.zeros_like(layer.W))
                self.vb.append(np.zeros_like(layer.b))

            self.initialized = True

        for i, layer in enumerate(layers):

            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.grad_b

            layer.W -= self.lr * self.vW[i]
            layer.b -= self.lr * self.vb[i]


class NAG:

    def __init__(self, lr=0.001, beta=0.9):

        self.lr = lr
        self.beta = beta

        self.vW = []
        self.vb = []

        self.initialized = False

    def step(self, layers):

        if not self.initialized:

            for layer in layers:

                self.vW.append(np.zeros_like(layer.W))
                self.vb.append(np.zeros_like(layer.b))

            self.initialized = True

        for i, layer in enumerate(layers):

            v_prev_W = self.vW[i]
            v_prev_b = self.vb[i]

            self.vW[i] = self.beta * self.vW[i] + self.lr * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + self.lr * layer.grad_b

            layer.W -= (self.beta * v_prev_W + (1 + self.beta) * (self.vW[i] - self.beta * v_prev_W))
            layer.b -= (self.beta * v_prev_b + (1 + self.beta) * (self.vb[i] - self.beta * v_prev_b))


class RMSProp:

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):

        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.sW = []
        self.sb = []

        self.initialized = False

    def step(self, layers):

        if not self.initialized:

            for layer in layers:

                self.sW.append(np.zeros_like(layer.W))
                self.sb.append(np.zeros_like(layer.b))

            self.initialized = True

        for i, layer in enumerate(layers):

            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)