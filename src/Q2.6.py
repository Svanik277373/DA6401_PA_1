import sys
import train

losses = ["cross_entropy", "mse"]

for loss in losses:

    sys.argv = [
        "train.py",
        "-o", "rmsprop",
        "-l", loss,
        "-a", "relu",
        "-sz", "128", "128", "128",
        "-lr", "0.001",
        "-e", "5",
        "-b", "128"
    ]

    print(f"\nRunning with loss: {loss}\n")

    train.main()