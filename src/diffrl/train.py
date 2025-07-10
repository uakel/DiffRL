from os import environ
from tqdm import tqdm

from functools import partial

from brax.training.agents.apg.train import train
from brax.envs import get_environment

import diffrl.envs.arm26

ENV_NAME = "arm26"
N_TRAINING_ITERS = 10

train_apg = partial(
    train,
    episode_length=1000,
    policy_updates=50,
)

def progress_fn(num_steps, metrics):
    print(num_steps, metrics)

environment = get_environment(ENV_NAME)

inference_fn, params, _ = train_apg(
    environment=environment,
    progress_fn=progress_fn,
)

print(
    inference_fn,
    params,
    _
)
