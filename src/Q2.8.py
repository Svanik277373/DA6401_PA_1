import numpy as np
import wandb
import argparse

from utils.data_loader import DataLoader
from ann.neural_network import NeuralNetwork




def load_model(model_path):

    args = argparse.Namespace(
        hidden_size=[128,128,128],
        activation="relu",
        weight_init="xavier",
        loss="cross_entropy",
        optimizer="rmsprop",
        learning_rate=0.001
    )

    model = NeuralNetwork(args)

    weights = np.load(model_path, allow_pickle=True).item()

    for i, layer in enumerate(model.layers):

        layer.W = weights[f"W{i}"]
        layer.b = weights[f"b{i}"]

    return model




def compute_confusion_matrix(y_true, y_pred, num_classes=10):

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm




def main():

    wandb.init(project="da6401_assignment1", name="error_analysis")

    loader = DataLoader("fashion_mnist")

    X_train, y_train, X_test, y_test = loader.load_data()

    model = load_model("best_model.npy")

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

   

    class_names = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]

    wandb.log({
        "confusion_matrix":
        wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=preds,
            class_names=class_names
        )
    })



    misclassified = np.where(preds != labels)[0]

    images = []

    for i in misclassified[:20]:

        img = X_test[i].reshape(28,28)

        caption = f"True:{class_names[labels[i]]} Pred:{class_names[preds[i]]}"

        images.append(wandb.Image(img, caption=caption))

    wandb.log({"misclassified_examples": images})

    wandb.finish()


if __name__ == "__main__":
    main()