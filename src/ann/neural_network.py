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

        num_layers = getattr(cli_args, "num_layers", 2)
        hidden_sizes = getattr(cli_args, "hidden_size", [128]*num_layers)

        if len(hidden_sizes) < num_layers:
            hidden_sizes = hidden_sizes + [hidden_sizes[-1]]*(num_layers-len(hidden_sizes))

        activation = getattr(cli_args, "activation", "relu")
        weight_init = getattr(cli_args, "weight_init", "xavier")

        loss_name = getattr(cli_args, "loss", "cross_entropy")
        optimizer_name = getattr(cli_args, "optimizer", "sgd")
        lr = getattr(cli_args, "learning_rate", 0.001)

        sizes = [input_dim] + hidden_sizes[:num_layers] + [output_dim]

        for i in range(len(sizes)-1):

            act = activation if i < len(sizes)-2 else None

            layer = NeuralLayer(
                sizes[i],
                sizes[i+1],
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

        out = X

        for i, layer in enumerate(self.layers):

            out = layer.forward(out)

            if hasattr(self.optimizer, "lr") and self.optimizer.lr >= 0.1:

                dead_fraction = np.mean(out == 0)

                log_metrics({
                    f"layer_{i}_dead_fraction": dead_fraction
                })

        return out

    def backward(self, X=None, y=None):

        if X is not None and y is not None:

            logits = self.forward(X)
            self.loss.forward(y, logits)

        grad = self.loss.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self):

        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1, batch_size=32):

        n = X_train.shape[0]

        metric_subset = 500
        iteration = 0

        for epoch in range(epochs):

            perm = np.random.permutation(n)

            for i in range(0, n, batch_size):

                idx = perm[i:i+batch_size]

                X_batch = X_train[idx]
                y_batch = y_train[idx]

                logits = self.forward(X_batch)

                loss_val = self.loss.forward(y_batch, logits)

                self.backward()

                if iteration < 50:

                    grad_matrix = self.layers[0].grad_W
                    grad_logs = {}

                    for neuron in range(min(5, grad_matrix.shape[1])):

                        grad_logs[f"grad_neuron_{neuron}"] = float(
                            np.mean(np.abs(grad_matrix[:, neuron]))
                        )

                    log_metrics(grad_logs)

                grad_norm = np.linalg.norm(self.layers[0].grad_W)

                self.update_weights()

                iteration += 1

                grad_norm_layer1 = grad_norm

            train_logits = self.forward(X_train[:metric_subset])
            train_loss = self.loss.forward(y_train[:metric_subset], train_logits)

            train_preds = np.argmax(train_logits, axis=1)
            train_labels = np.argmax(y_train[:metric_subset], axis=1)

            train_accuracy = np.mean(train_preds == train_labels)

            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "gradient_norm": grad_norm,
                "grad_norm_layer1": grad_norm_layer1
            }

            if X_val is not None:

                val_logits = self.forward(X_val[:metric_subset])
                val_loss = self.loss.forward(y_val[:metric_subset], val_logits)

                val_preds = np.argmax(val_logits, axis=1)
                val_labels = np.argmax(y_val[:metric_subset], axis=1)

                val_accuracy = np.mean(val_preds == val_labels)

                metrics["val_loss"] = val_loss
                metrics["val_accuracy"] = val_accuracy

            log_metrics(metrics)

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

            key_w = f"W{i}"
            key_b = f"b{i}"

            if key_w in weights:
                layer.W = weights[key_w]

            if key_b in weights:
                layer.b = weights[key_b]
