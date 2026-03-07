"""
Utility wrapper for Weights & Biases logging
"""

import wandb


def init_wandb(project, config=None, group=None):

    wandb.init(
        project=project,
        config=config,
        group=group,
        settings=wandb.Settings(init_timeout=300)
    )

def log_metrics(metrics):

    if wandb.run is not None:
        wandb.log(metrics)
    else:
        print("W&B run not initialized. Metrics:", metrics)


def finish_wandb():

    if wandb.run is not None:
        wandb.finish()