"""
Inference script
"""

import argparse
import numpy as np

from utils.data_loader import DataLoader
from ann.neural_network import NeuralNetwork
from ann.objective_functions import Loss


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="best_model.npy")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-b", "--batch_size", type=int, default=64)

    parser.add_argument("-nhl", "--num_layers", type=int, default=3)

    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+",
                        default=[128, 128, 128])

    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    return parser.parse_args()


def load_model(model_path):

    data = np.load(model_path, allow_pickle=True).item()
    return data


def set_model_weights(model, weights):

    for i, layer in enumerate(model.layers):

        layer.W = weights[f"W{i}"]
        layer.b = weights[f"b{i}"]


def evaluate_model(model, X_test, y_test):

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(preds == labels)

    
    num_classes = 10

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):

        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return results


def main():

    args = parse_arguments()

    model_path = args.model_path
    dataset = args.dataset

    weights = load_model(model_path)

    print("Loaded model:", model_path)
    print("Dataset:", dataset)

    
    loader = DataLoader(dataset)

    X_train, y_train, X_test, y_test = loader.load_data()

    
    class Args:
        hidden_size = args.hidden_size
        activation = args.activation
        learning_rate = 0.001
        optimizer = "sgd"
        loss = "cross_entropy"
        weight_init = "xavier"

    model = NeuralNetwork(Args)

   
    set_model_weights(model, weights)

    
    results = evaluate_model(model, X_test, y_test)

    print("\nEvaluation Results")
    print("------------------")
    print("Accuracy :", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall   :", results["recall"])
    print("F1 Score :", results["f1"])


if __name__ == "__main__":
    main()