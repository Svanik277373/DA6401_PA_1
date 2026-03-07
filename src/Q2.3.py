import sys
import train

optimizers = ["sgd", "momentum", "nag", "rmsprop"]

for opt in optimizers:

    sys.argv = [
        "train.py",
        "-o", opt,
        "-e", "5",
        "-sz", "128", "128", "128",
        "-a", "relu",
        "-b", "128",
        "-lr", "0.001",
        "-wp", "da6401_assignment1"
    ]

    print(f"\nRunning optimizer: {opt}\n")

    train.main()