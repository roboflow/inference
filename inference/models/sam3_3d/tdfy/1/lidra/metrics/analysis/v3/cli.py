"""Minimal Hydra CLI v2 - direct instantiation only."""

import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from loguru import logger

# Register custom resolvers
OmegaConf.register_new_resolver(
    "rv.if_not_none",
    lambda value, if_not_none, if_none: if_not_none if value is not None else if_none,
    replace=True,
)

# Get the absolute path to the config directory
_FILE_DIR = Path(__file__).parent
_CONFIG_PATH = (_FILE_DIR / "../../../../etc/lidra/run/analysis").resolve()


@hydra.main(
    version_base="1.3",
    config_path=str(_CONFIG_PATH),
    config_name="multi_trial/stage1",  # Default to stage1 config
)
def main(cfg: DictConfig) -> int:
    """
    Minimal metrics analysis CLI using direct Hydra instantiation.

    Expects configs with:
    - processor._target_: processor class to instantiate
    - config._target_: config class to instantiate
    """
    # Log the config name for debugging
    config_name = hydra.core.hydra_config.HydraConfig.get().job.config_name
    logger.success(f"Using config: {config_name}")
    config = hydra.utils.instantiate(cfg.config)
    processor = hydra.utils.instantiate(cfg.processor)
    processor.run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
