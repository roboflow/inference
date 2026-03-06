import os
from typing import List
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

# Register custom resolvers
OmegaConf.register_new_resolver(
    "rv.if_not_none",
    lambda value, if_not_none, if_none: if_not_none if value is not None else if_none,
    replace=True,
)


def _detect_rank_and_world_size() -> tuple[int, int]:
    """Detect global rank and world size from common env vars.

    Priority: RANK/WORLD_SIZE -> SLURM_PROCID/SLURM_NTASKS -> defaults.
    """
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    return rank, world_size


class ShardedDataset:
    """Wrap a dataset to expose only a shard for the given rank/world size."""

    def __init__(self, base_dataset, rank: int, world_size: int):
        self.base_dataset = base_dataset
        self.rank = rank
        self.world_size = max(1, world_size)
        # Precompute the indices served by this rank for stable ordering
        total = len(base_dataset)
        self.indices: List[int] = [
            i for i in range(total) if (i % self.world_size) == self.rank
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.indices[index]]


def _instantiate_dataset(cfg: DictConfig):
    """Instantiate the dataset defined in cfg.dataset."""
    dataset = instantiate(cfg.dataset)
    total_len = len(dataset)

    # Partition dataset across processes for SLURM/submitit runs
    rank, world_size = _detect_rank_and_world_size()
    if world_size > 1:
        logger.info(
            f"Sharding dataset across {world_size} ranks. (Process rank: {rank})"
        )
        dataset = ShardedDataset(dataset, rank=rank, world_size=world_size)
        logger.info(
            f"Dataset size: total={total_len}, shard[{rank}/{world_size}]={len(dataset)}"
        )
    else:
        logger.info(f"Dataset size: total={total_len}")
    return dataset


def _set_cuda_visible_devices():
    # Ensure each local process is pinned to a single GPU via CUDA_VISIBLE_DEVICES
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    # Lazily import to avoid heavy deps until needed
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def main_fn(cfg: DictConfig) -> None:
    _set_cuda_visible_devices()

    # Instantiate dataset defined in cfg.dataset
    if "dataset" not in cfg:
        raise ValueError(
            "cfg.dataset is required (Hydra _target_) to instantiate a visualization dataset"
        )
    dataset = _instantiate_dataset(cfg)

    from scripts.weiyaowang.visualization.inference_pipeline_visualization import (
        visualize_results,
    )

    # Avoid output collisions: write into per-rank subfolders when distributed
    output_dir = str(cfg.visualization.output_dir)

    visualize_results(
        demo_version=str(cfg.demo_version),
        dataset=dataset,
        output_dir=output_dir,
        demo_config_override_func=None,
        inferenece_pipeline_config_modification_func=None,
        viz_single_obj=bool(cfg.visualization.viz_single_obj),
        stage2_only=bool(cfg.visualization.stage2_only),
        export_obj_gif=bool(cfg.visualization.export_obj_gif),
        viz_pm=bool(cfg.visualization.viz_pm),
    )

    logger.info(f"Visualization complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    run(main_fn, config_name="run/visualization/base.yaml")
