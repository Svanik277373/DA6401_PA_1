import wandb
import multiprocessing
import sys
import os

from train import main

# Prevent BLAS thread explosion
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT = "da6401_assignment1"

sweep_config = {

    "method": "grid",

    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },

    "parameters": {

        "epochs": {"values": [1]},

        "batch_size": {"values": [128]},

        "learning_rate": {"values": [0.001]},

        "optimizer": {"values": ["sgd"]},

        "activation": {"values": ["relu"]},

        "loss": {"values": ["cross_entropy"]},

        "hidden_size": {"values": [[128,128,128]]},

        # Compare these two initializations
        "weight_init": {"values": ["zeros", "xavier"]}
    }
}


def run_sweep():

    run = wandb.init()
    config = wandb.config

    arch_str = "_".join(map(str, config.hidden_size))
    run.name = f"{config.weight_init}_init_{arch_str}"

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

    wandb.agent(
        sweep_id,
        function=run_sweep,
        count=1   # one run per agent
    )


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project=PROJECT)

    print("Sweep ID:", sweep_id)

    processes = []

    NUM_AGENTS = 2   # two parallel runs (zeros, xavier)

    for _ in range(NUM_AGENTS):

        p = multiprocessing.Process(
            target=start_agent,
            args=(sweep_id,)
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()