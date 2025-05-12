import os

import torch

from inference.v1.utils.environment import parse_comma_separated_values

ONNXRUNTIME_EXECUTION_PROVIDERS = parse_comma_separated_values(
    values=os.getenv(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider",
    )
)
DEFAULT_DEVICE = torch.get_default_device()
