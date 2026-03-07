"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist


class DataLoader:

    def __init__(self, dataset="mnist"):

        if dataset not in ["mnist", "fashion_mnist"]:
            raise ValueError("Dataset must be mnist or fashion_mnist")

        self.dataset = dataset

    def load_data(self):

        if self.dataset == "mnist":
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

        else:
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # normalize
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # flatten images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # one-hot encode labels
        y_train = self.one_hot(y_train, 10)
        y_test = self.one_hot(y_test, 10)

        return X_train, y_train, X_test, y_test

    def one_hot(self, y, num_classes):

        one_hot = np.zeros((y.shape[0], num_classes))

        one_hot[np.arange(y.shape[0]), y] = 1

        return one_hot