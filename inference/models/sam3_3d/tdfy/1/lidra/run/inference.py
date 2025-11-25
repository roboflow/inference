from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from lidra.run.common import run
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


def inference_fn(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config, resolve=True))

    # run inference
    loop = instantiate(config.loop)
    kwargs = instantiate(config.predict.args)
    loop.log_config(config)
    loop.predict(**kwargs)


if __name__ == "__main__":
    run(inference_fn, config_name="run/eval/default.yaml")
