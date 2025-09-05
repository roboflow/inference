# from .v1 import HfMoondream as HfMoondreamV1

from inference_exp.models.moondream2.transformers_model.v1.hf_moondream import (
    HfMoondream as HfMoondreamV1,
)


def get_model_class():
    return HfMoondreamV1


# e.g. if we need different model classes for different transformers versions, we can do:
"""
from .v1.hf_moondream import HfMoondream as HfMoondreamV1
from .v2.hf_moondream import HfMoondream as HfMoondreamV2
import transformers
from packaging import version

def get_model_class():
    transformers_version = transformers.__version__
    if version.parse(transformers_version) > version.parse("0.XX.X"):
        return HfMoondreamV2
    else:
        return HfMoondreamV1
"""
