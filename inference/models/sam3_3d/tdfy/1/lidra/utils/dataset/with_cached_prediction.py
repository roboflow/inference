#!/usr/bin/env python3
"""
Simple function to instantiate a WithCachedPredictionDataset from a cached predictions directory.

This loads the cached predictions along with the original dataset configuration
that was used to generate them.
"""

from pathlib import Path
from typing import Optional
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Import and register resolvers before any config loading
import lidra.init.resolvers  # This registers all custom resolvers

from lidra.data.dataset.tdfy.with_cached_predictions_dataset import (
    WithCachedPredictionDataset,
)
from lidra.utils.notebook.hydra import isolated_hydra_context
import lidra
from loguru import logger


def load_cached_predictions_dataset(
    prediction_folder: str,
    dataset_config_key: str = "loop.dataloaders.validation.dataset",
    device: torch.device = torch.device("cpu"),
) -> WithCachedPredictionDataset:
    """
    Load a WithCachedPredictionDataset from a cached predictions folder.

    This function:
    1. Loads the original experiment config from the parent directory
    2. Extracts and resolves the validation dataset configuration
    3. Instantiates the original dataset
    4. Wraps it with WithCachedPredictionDataset to include predictions

    Args:
        prediction_folder: Path to the cached_predictions directory
                         (e.g., /checkpoint/3dfy/user/lidra/logs/tagged/_submit/exp.preds.r3_anything/cached_predictions)
        dataset_config_key: Key path to dataset config in the config file (default: 'loop.dataloaders.validation.dataset')
        device: Device to load tensors to (default: cpu)

    Returns:
        WithCachedPredictionDataset instance with both samples and predictions
    """
    prediction_folder = Path(prediction_folder)
    if not prediction_folder.exists():
        raise ValueError(f"Prediction folder does not exist: {prediction_folder}")

    # The config should be in the parent directory (the .preds.* directory)
    config_file_path = prediction_folder.parent / "config.yaml"
    if not config_file_path.exists():
        raise ValueError(f"Config file not found at: {config_file_path}")

    logger.info(f"Loading config from: {config_file_path}")

    # Load the full config without resolving (to avoid hydra runtime dependencies)
    full_config = OmegaConf.load(config_file_path)
    dataset_config = OmegaConf.select(full_config, dataset_config_key)
    if dataset_config is None:
        raise ValueError(
            f"Could not find dataset configuration at '{dataset_config_key}' or alternative paths in config file"
        )

    OmegaConf.resolve(dataset_config)

    # Get the lidra config root for hydra initialization
    lidra_root = Path(lidra.__file__).parent.parent / "etc" / "lidra"

    # Instantiate the sample dataset using isolated hydra context
    logger.info("Instantiating sample dataset...")
    with isolated_hydra_context(lidra_root):
        sample_dataset = instantiate(dataset_config)

    logger.info(f"Sample dataset has {len(sample_dataset)} samples")

    # Create the WithCachedPredictionDataset
    dataset = WithCachedPredictionDataset(
        sample_dataset=sample_dataset,
        prediction_folder=prediction_folder,
        device=device,
    )

    logger.info(f"Loaded {len(dataset.prediction_metadata)} cached predictions")
    return dataset
