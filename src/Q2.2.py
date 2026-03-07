import wandb
import multiprocessing
import sys
import os

from train import main

# Prevent NumPy / BLAS from using multiple threads
# Improves performance when running parallel sweeps
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT = "da6401_assignment1"

HIDDEN_OPTIONS = [
    [64],
    [128],
    [64,64],
    [128,64],
    [128,128],
    [64,64,64],
    [128,64,64],
    [128,128,64],
    [128,128,128]
]

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {

        "epochs": {"values": [5,10]},

        "batch_size": {"values": [32,64,128]},

        "learning_rate": {"values": [1e-4,5e-4,1e-3,5e-3,1e-2]},

        "optimizer": {"values": ["sgd","momentum","nag","rmsprop"]},

        "activation": {"values": ["relu","tanh","sigmoid"]},

        "loss": {"values": ["cross_entropy","mse"]},

        "weight_init": {"values": ["random","xavier"]},

        "hidden_size": {"values": HIDDEN_OPTIONS}
    }
}


def run_sweep():
    """
    Runs one training job using sweep hyperparameters.
    """

    run = wandb.init()
    config = wandb.config

    # Create readable architecture string
    arch_str = "_".join(map(str, config.hidden_size))

    # Assign clear run name based on hyperparameters
    run.name = (
        f"{config.optimizer}"
        f"_lr{config.learning_rate}"
        f"_bs{config.batch_size}"
        f"_act{config.activation}"
        f"_h{arch_str}"
    )

    # Convert sweep config → CLI arguments
    args = [
        "train.py",
        "-e", str(config.epochs),
        "-b", str(config.batch_size),
        "-lr", str(config.learning_rate),
        "-o", config.optimizer,
        "-a", config.activation,
        "-l", config.loss,
        "-wi", config.weight_init,
        "-sz", *map(str, config.hidden_size)
    ]

    sys.argv = args

    main()


def start_agent(sweep_id):
    """
    Each process runs a W&B agent executing multiple runs.
    """
    wandb.agent(sweep_id, function=run_sweep, count=20)


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project=PROJECT)

    print("Sweep ID:", sweep_id)

    processes = []

    NUM_AGENTS = 5   # number of parallel workers

    for _ in range(NUM_AGENTS):

        p = multiprocessing.Process(
            target=start_agent,
            args=(sweep_id,)
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()