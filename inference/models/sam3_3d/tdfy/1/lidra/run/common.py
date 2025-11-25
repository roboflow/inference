import os
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from lidra.config.utils import dump_config
from lidra.init.environment import init_env_variables


def get_main_filename():
    import __main__

    path = __main__.__file__
    name = os.path.basename(path)
    return os.path.splitext(name)[0]


def should_symlink(dst, name, overwrite):
    if os.path.islink(dst):
        logger.warning(f"tag path '{dst}' already exist")
        if overwrite:
            logger.warning(f"overwriting it")
            os.unlink(dst)
            return True
        # occur when restarting python process in same folder
        elif os.path.samefile(dst, os.getcwd()):
            return False
        else:
            raise RuntimeError(
                f"cannot overwrite tag '{name}', add option 'tag.overwrite=true' to allow overwriting"
            )
    return True


def attempt_tag_link(config: DictConfig):
    from lightning.pytorch.utilities.rank_zero import (
        rank_zero_only,
    )  # indirectly load the `transformers` module, which messes up the os.environ["HF_HOME"] path, had to be moved there

    @rank_zero_only
    def run_on_rank_zero(config: DictConfig):
        if ("tag" not in config) or (config.tag.name is None):
            logger.warning(f"no tag name has been provided, skipping ...")
            logger.warning(
                f"tags can be added by passing the argument 'tag.name=<name>'"
            )
        else:
            tag_name = f"{get_main_filename()}/{config.tag.name}"
            tag_root = os.path.join(os.getcwd().split("timestamped")[0], "tagged")
            tag_dst = os.path.join(tag_root, tag_name)
            tag_dir = os.path.dirname(tag_dst)
            os.makedirs(tag_dir, exist_ok=True)

            if should_symlink(tag_dst, config.tag.name, config.tag.overwrite):
                # create tag link
                os.symlink(
                    os.path.relpath(
                        os.getcwd(),
                        tag_dir,
                    ),
                    tag_dst,
                )
            logger.info(f"tags '{config.tag.name}' has been created here : {tag_dst}")

    run_on_rank_zero(config)


def run_main_with_body(body_fn, config_processing_fn=None) -> None:
    def run_main(config: DictConfig) -> None:
        # save config file (without the hydra section)
        logger.info(OmegaConf.to_yaml(config, sort_keys=True))

        dump_config(config)

        logger.info(f"saving logs, configs, and model checkpoints to {os.getcwd()}")

        # optional processing of the config structure
        if config_processing_fn is not None:
            config = config_processing_fn(config)

        # init procedures
        init_env_variables(config)

        attempt_tag_link(config)

        try:
            body_fn(config)
        except Exception as e:
            logger.opt(exception=True).error("exception occuring while training")
            raise e from e

        logger.info(f"training ran successfully, logs can be found in : {os.getcwd()}")

    # this line is to fix issue with hydra `config_path``
    run_main.__module__ = body_fn.__module__

    return run_main


def run(body_fn, config_processing_fn=None, **hydra_kwargs):
    # wrap main with hydra stuff
    main_fn = hydra.main(
        version_base="1.3",
        config_path="../../etc/lidra",
        **hydra_kwargs,
    )(run_main_with_body(body_fn, config_processing_fn))
    main_fn()
