import os
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lidra.run.common import run
from lidra.model.io import get_last_checkpoint

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


def resume_fn(config: DictConfig):
    loop = instantiate(config.loop)
    loop.fit()


def config_processing_fn(config: DictConfig):
    logs_path = os.path.abspath(config.logs.path)
    os.chdir(logs_path)
    training_config = OmegaConf.load("config.yaml")
    training_config.loop["checkpoint"] = get_last_checkpoint(
        "lightning/version_0/checkpoints"
    )  # "last"
    training_config.loop.logger.version = 0
    return training_config


if __name__ == "__main__":
    run(resume_fn, config_processing_fn, config_name="run/train/resume.yaml")
