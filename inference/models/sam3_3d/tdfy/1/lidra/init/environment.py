import os
import sys
from omegaconf import DictConfig


def init_env_variables(config: DictConfig):
    # huggingface cache directory
    if "cluster" in config:
        os.environ["HF_HOME"] = os.path.join(config.cluster.path.cache, "hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_python_path():
    current_file_path = os.path.join(os.getcwd(), __file__)
    current_dir_path = os.path.dirname(current_file_path)
    dependencies_path = os.path.join(
        current_dir_path,
        "..",
        "..",
        "external",
        "dependencies",
    )
    dependencies_path = os.path.abspath(dependencies_path)

    # add external dependencies
    sys.path.insert(0, dependencies_path)
