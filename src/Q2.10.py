import wandb
import multiprocessing
import sys
import os

from train import main

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

        "dataset": {"values": ["fashion_mnist"]},

        "epochs": {"values": [10]},

        "batch_size": {"values": [128]},

        "learning_rate": {"values": [0.001]},

        "loss": {"values": ["cross_entropy"]},

        "weight_init": {"values": ["xavier"]},

        "config_id": {"values": [1,2,3]}
    }
}


def run_sweep():

    run = wandb.init()
    config = wandb.config

    if config.config_id == 1:

        hidden = [128,128,128]
        optimizer = "rmsprop"
        activation = "relu"

    elif config.config_id == 2:

        hidden = [256,128,64]
        optimizer = "momentum"
        activation = "relu"

    else:

        hidden = [128,128]
        optimizer = "rmsprop"
        activation = "tanh"

    run.name = f"config_{config.config_id}"

    args = [
        "train.py",
        "-d", "fashion_mnist",
        "-e", str(config.epochs),
        "-b", str(config.batch_size),
        "-lr", str(config.learning_rate),
        "-o", optimizer,
        "-a", activation,
        "-l", config.loss,
        "-wi", config.weight_init,
        "-sz", *map(str, hidden)
    ]

    sys.argv = args

    main()


def start_agent(sweep_id):

    wandb.agent(sweep_id, function=run_sweep, count=1)


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project=PROJECT)

    processes = []

    NUM_AGENTS = 3

    for _ in range(NUM_AGENTS):

        p = multiprocessing.Process(
            target=start_agent,
            args=(sweep_id,)
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()