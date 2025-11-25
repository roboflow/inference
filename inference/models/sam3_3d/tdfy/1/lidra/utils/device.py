"""Device utilities for distributed training."""

import os
import torch
from loguru import logger


def get_auto_device(device=None):
    """Get the appropriate device based on environment variables for distributed training.

    Args:
        device: Device specification. Can be:
            - None or "auto": Auto-detect based on environment (uses LOCAL_RANK/SLURM_LOCALID)
            - "cuda": Auto-select GPU based on local rank
            - "cuda:N": Use specific GPU N (returned as-is)
            - "cpu": Use CPU

    Returns:
        Device string to use (e.g., "cuda:0", "cuda:1", "cpu")
    """
    if device is None or device == "auto" or device == "cuda":
        if torch.cuda.is_available():
            try:
                # Try SLURM first
                local_rank = int(os.environ.get("SLURM_LOCALID"))
                device_str = f"cuda:{local_rank}"
                logger.info(f"Auto-detected device from SLURM_LOCALID: {device_str}")
                return device_str
            except (TypeError, ValueError):
                try:
                    # Try standard LOCAL_RANK
                    local_rank = int(os.environ.get("LOCAL_RANK"))
                    device_str = f"cuda:{local_rank}"
                    logger.info(f"Auto-detected device from LOCAL_RANK: {device_str}")
                    return device_str
                except (TypeError, ValueError):
                    # Default to cuda:0 if no distributed env vars
                    if device == "cuda":
                        logger.info("No distributed environment detected, using cuda:0")
                        return "cuda:0"
                    else:
                        logger.info("No distributed environment detected, using cuda")
                        return "cuda"
        else:
            logger.info("CUDA not available, using cpu")
            return "cpu"

    # Return the device as-is if it's a specific device string
    return device
