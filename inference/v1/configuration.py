import os

import torch

from inference.v1.utils.environment import parse_comma_separated_values

ONNXRUNTIME_EXECUTION_PROVIDERS = parse_comma_separated_values(
    values=os.getenv(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider",
    )
)
DEFAULT_DEVICE_STR = os.getenv(
    (
        "DEFAULT_DEVICE" "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
)
DEFAULT_DEVICE = torch.device(DEFAULT_DEVICE_STR)
