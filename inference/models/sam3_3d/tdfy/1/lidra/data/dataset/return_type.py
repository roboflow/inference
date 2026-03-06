from collections import namedtuple
from typing import Any
import torch

AbstractDatasetReturnType = namedtuple(
    "AbstractDatasetReturnType", ["sample_uuid", "data"]
)


def extract_data(x: AbstractDatasetReturnType) -> Any:
    return x[1]


def extract_sample_uuid(x: AbstractDatasetReturnType) -> Any:
    return x[0]


class SampleUuidUtils:
    @staticmethod
    def demote(x: Any) -> str:
        if isinstance(x, str):
            return x
        elif isinstance(x, int):
            return x
        elif isinstance(x, (tuple, list)):
            return tuple(SampleUuidUtils.demote(y) for y in x)
        elif isinstance(x, torch.Tensor):
            x = x.tolist()
            if isinstance(x, list):
                return tuple(SampleUuidUtils.demote(y) for y in x)
            else:
                return x
        else:
            raise ValueError(f"Cannot demote {type(x)} to a primitive type {x}")
