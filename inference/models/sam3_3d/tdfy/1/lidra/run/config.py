from lidra.run.common import run, dump_config
from omegaconf import DictConfig
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


def config_fn(config: DictConfig):
    dump_config(config, config.output)


if __name__ == "__main__":
    run(config_fn)
