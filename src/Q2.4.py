import sys
import train

activations = ["sigmoid", "relu"]

architectures = [
    [128],
    [128,128],
    [128,128,128],
    [128,128,128,128]
]

for activation in activations:
    for arch in architectures:

        sys.argv = [
            "train.py",
            "-o","rmsprop",
            "-a",activation,
            "-sz",*map(str,arch),
            "-e","5",
            "-b","128"
        ]

        print(f"\nActivation: {activation} | Architecture: {arch}\n")

        train.main()