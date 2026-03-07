import numpy as np

from ann.neural_layer import NeuralLayer
from ann.objective_functions import Loss
from ann.optimizers import SGD, Momentum, NAG, RMSProp
from utils.wandb import log_metrics


class NeuralNetwork:

    def __init__(self, cli_args):

        self.layers = []

        input_dim = 784
        output_dim = 10

        hidden_sizes = getattr(cli_args, "hidden_size", [])
        activation = getattr(cli_args, "activation", "relu")
        weight_init = getattr(cli_args, "weight_init", "xavier")

        loss_name = getattr(cli_args, "loss", "cross_entropy")
        optimizer_name = getattr(cli_args, "optimizer", "sgd")
        lr = getattr(cli_args, "learning_rate", 0.001)

        sizes = [input_dim] + hidden_sizes + [output_dim]

        for i in range(len(sizes) - 1):

            act = activation if i < len(sizes) - 2 else None

            layer = NeuralLayer(
                sizes[i],
                sizes[i + 1],
                activation=act,
                weight_init=weight_init
            )

            self.layers.append(layer)

        self.loss = Loss(loss_name)

        if optimizer_name == "sgd":
            self.optimizer = SGD(lr)

        elif optimizer_name == "momentum":
            self.optimizer = Momentum(lr)

        elif optimizer_name == "nag":
            self.optimizer = NAG(lr)

        elif optimizer_name == "rmsprop":
            self.optimizer = RMSProp(lr)

    def forward(self, X):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        out = X

        for i, layer in enumerate(self.layers):
            out = layer.forward(out)

        return out

    def backward(self, X=None, y=None):

        if X is not None and y is not None:

            if X.ndim == 1:
                X = X.reshape(1, -1)

            if y.ndim == 1:
                y = y.reshape(1, -1)

            logits = self.forward(X)
            self.loss.forward(y, logits)

        grad = self.loss.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1, batch_size=32):

        n = X_train.shape[0]

        for epoch in range(epochs):

            perm = np.random.permutation(n)

            for i in range(0, n, batch_size):

                idx = perm[i:i + batch_size]

                X_batch = X_train[idx]
                y_batch = y_train[idx]

                logits = self.forward(X_batch)

                self.loss.forward(y_batch, logits)

                self.backward()

                self.update_weights()

    def evaluate(self, X, y):

        logits = self.forward(X)

        loss_val = self.loss.forward(y, logits)

        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)

        accuracy = np.mean(preds == labels)

        return loss_val, accuracy

    def get_weights(self):

        d = {}

        for i, layer in enumerate(self.layers):

            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()

        return d

    def set_weights(self, weights):

        for i, layer in enumerate(self.layers):

            if f"W{i}" in weights:
                layer.W = weights[f"W{i}"]

            if f"b{i}" in weights:
                layer.b = weights[f"b{i}"]
