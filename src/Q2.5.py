import sys
import train

configs = [
    ("relu", 0.1),
    ("tanh", 0.1)
]

for activation, lr in configs:

    sys.argv = [
        "train.py",
        "-o", "rmsprop",
        "-a", activation,
        "-sz", "128", "128", "128",
        "-e", "5",
        "-lr", str(lr),
        "-b", "128"
    ]

    print(f"\nActivation: {activation} | LR: {lr}\n")

    train.main()