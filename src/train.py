"""
Training script for the MLP model
"""

import argparse
import numpy as np

from utils.data_loader import DataLoader
from ann.neural_network import NeuralNetwork
from utils.wandb import init_wandb, log_metrics, finish_wandb


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=5)

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=128)

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.001)

    parser.add_argument("-o", "--optimizer",
                        default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("-l", "--loss",
                        default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    parser.add_argument("-sz", "--hidden_size",
                        nargs="+",
                        type=int,
                        default=[128, 128, 128])

    parser.add_argument("-a", "--activation",
                        default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("-wi", "--weight_init",
                    default="xavier",
                    choices=["random", "xavier", "zeros"])

    parser.add_argument("-wp", "--wandb_project",
                        default="da6401_assignment1")

    parser.add_argument("--model_save_path",
                        default="best_model.npy")
    parser.add_argument("--save_confusion_matrix",
                        action="store_true",
                        help="Log confusion matrix after training")

    return parser.parse_args()

def save_confusion_matrix(model, X, y):

    import wandb
    import numpy as np

    logits = model.forward(X)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y, axis=1)

    class_names = [str(i) for i in range(10)]

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=preds,
            y_true=labels,
            class_names=class_names
        )
    })

def compute_f1(y_true, y_pred, num_classes=10):

    f1_scores = []

    for c in range(num_classes):

        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        f1_scores.append(f1)

    return np.mean(f1_scores)
def main():

    args = parse_arguments()

    init_wandb(args.wandb_project, vars(args))

    loader = DataLoader(args.dataset)

    X_train, y_train, X_test, y_test = loader.load_data()


    val_size = int(0.1 * len(X_train))

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    model = NeuralNetwork(args)

    
    model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    logits = model.forward(X_val)

    val_preds = np.argmax(logits, axis=1)
    val_labels = np.argmax(y_val, axis=1)

    val_f1 = compute_f1(val_labels, val_preds)
    log_metrics({
    "val_loss": float(val_loss),
    "val_accuracy": float(val_accuracy),
    "f1": float(val_f1)
    })

    import wandb
    wandb.run.summary["val_accuracy"] = float(val_accuracy)
    wandb.run.summary["val_loss"] = float(val_loss)
    wandb.run.summary["f1"] = float(val_f1)


    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    log_metrics({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    })

    if args.save_confusion_matrix:
        save_confusion_matrix(model, X_test, y_test)

    weights = model.get_weights()

    np.save(args.model_save_path, weights)

    finish_wandb()


if __name__ == "__main__":
    main()