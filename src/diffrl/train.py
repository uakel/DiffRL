from rich import print as rprint
from rich.pretty import Pretty

from omegaconf import OmegaConf
from tqdm import tqdm

from brax.training.agents.apg.train import train
from brax.envs import get_environment

# Constants
STANDARD_CONFIG_DICT = {
    "training": {
        "num_envs": 1,
        "num_evals": 32,
        "episode_length": 1000,
        "policy_updates": 1000,
    },
    "environment": "arm26",
    "wandb_project": "DiffRL",
}

# Environment Loader
def load_environment(
    env_name: str
):
    match env_name:
        case "arm26":
            import diffrl.envs.arm26
        case _:
            raise ValueError(
                "Provided environment not supported"
            )

# Logging function
def progress_fn(num_steps: int, metrics: dict):
    wandb.log(metrics)
    rprint(Pretty(metrics))
    rprint(Pretty(num_steps))

# Main
if __name__ == "__main__":
    # Configure cli interface
    from argparse import ArgumentParser
    import pprint
    parser = ArgumentParser()
    parser.add_argument(
        "config_file",
        nargs="?",
        help=f"Optional path to a .yaml config file "
             f"with training arguments. The file should "
             f"contain keys that override the defaults "
             f"in STANDARD_TRAIN_DICT. "
             f"Defaults:\n{pprint.pformat(STANDARD_CONFIG_DICT)}"
    )
    args = parser.parse_args()
    configuration = OmegaConf.create(STANDARD_CONFIG_DICT)

    # Update standard config if config is given
    if args.config_file is not None:
        configuration = OmegaConf.merge(
            configuration,
            OmegaConf.load(args.config_file)
        )

    # Load environment
    load_environment(configuration.environment)
    environment = get_environment(configuration.environment)

    # Logging
    import wandb
    wandb.login()
    wandb.init(
        project=configuration.wandb_project,
        config=OmegaConf.to_container( # type: ignore
            configuration, resolve=True, 
            throw_on_missing=True
        )
    )

    # Make training dict
    training_parameter_dict = OmegaConf.to_container(
        configuration.training, resolve=True
    )

    # Start training
    inference_fn, params, _ = train(
        environment,
        progress_fn=progress_fn,
        **training_parameter_dict # type: ignore
    )

