import os
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from loguru import logger

import lidra


class LidraConf:
    """
    This is largely a convenience class, providing a single way to load any lidra config in a notebook, without having to remember when to use OmegaConf.load or

    ```python
        # Avoid nested Hydra initialization errors when called from an app already using Hydra
        with isolated_hydra_context(Path(compose_root)):
            hydra_cfg = compose(config_name=config_path, overrides=overrides)
    ```

    when to use .yaml, at the end, etc
    """

    # Sasha: This is ugly but it's just for notebooks so ¯\_(ツ)_/¯
    LIDRA_CONF_ROOT = str(Path(lidra.__file__).parent.parent / "etc" / "lidra")

    @staticmethod
    def load_config(
        config_path: str,
        overrides: Optional[List[str]] = None,
        as_root: bool = True,
    ) -> DictConfig:
        if overrides is None:
            overrides = []

        if os.path.isabs(config_path):
            logger.warning(f"config_path is an absolute path: using OmegaConf.load")
            return LidraConf._initialize_config_omegaconf(
                config_path, overrides=overrides
            )
        else:
            logger.warning(f"config_path is not an absolute path: {config_path}")
            logger.warning(f"Loading with compose root @ {LidraConf.LIDRA_CONF_ROOT}")
            config = LidraConf._initialize_config_hydra(
                config_path,
                compose_root=LidraConf.LIDRA_CONF_ROOT,
                overrides=overrides,
            )
            if as_root:
                return LidraConf._warn_and_select_path_from_config(
                    config, config_path.split("/")[:-1]
                )
        return config

    @staticmethod
    def _warn_and_select_path_from_config(
        config: DictConfig, path: List[str]
    ) -> DictConfig:
        logger.warning(f"Selecting root config from {path}")
        config = OmegaConf.select(config, ".".join(path))
        if config is None:
            raise ValueError(f"Config at path {path} is None")
        return config

    @staticmethod
    def _initialize_config_hydra(
        config_path: str,
        compose_root: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> DictConfig:
        assert config_path.endswith(".yaml")  # For compatibility with cli
        config_path = config_path.replace(".yaml", "")

        if compose_root is None:
            compose_root = str(LidraConf.LIDRA_CONF_ROOT)

        with isolated_hydra_context(Path(composite_root := compose_root)):
            hydra_cfg = compose(config_name=config_path, overrides=overrides)

        return hydra_cfg

    def _initialize_config_omegaconf(
        config_path: str,
        overrides: Optional[List[str]] = None,
    ) -> DictConfig:
        config = OmegaConf.load(config_path)
        override_cfg = OmegaConf.from_dotlist(overrides)
        return OmegaConf.merge(config, override_cfg)

    @staticmethod
    def instantiate_and_print(config, *args, **kwargs):
        logger.info(OmegaConf.to_yaml(config))
        instantiated = instantiate(config, *args, **kwargs)
        return instantiated


@contextmanager
def isolated_hydra_context(config_dir: Path):
    """Context manager for isolated Hydra execution.

    Mirrors the implementation used in analysis processors to safely
    create a nested Hydra context. Saves and restores Hydra state so
    callers inside an existing Hydra app can compose configs.
    """
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir

    # Save current state
    is_initialized = GlobalHydra.instance().is_initialized()
    saved_hydra = GlobalHydra.instance().hydra if is_initialized else None

    # Clear for new context
    if is_initialized:
        GlobalHydra.instance().clear()

    try:
        # Initialize new context
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            yield
    finally:
        # Restore original state
        if is_initialized and saved_hydra:
            GlobalHydra.instance().initialize(saved_hydra)


def replace_from_path(
    dict_config: DictConfig,
    val_fpath: str,
    key: str,
    override_keys: Optional[List[str]] = None,
):
    """
    Replace the config at the given path with the new config.
    """
    old_config = dict_config.copy()
    new_config = LidraConf.load_config(val_fpath, override_keys)
    old_config[key] = new_config
    return old_config
