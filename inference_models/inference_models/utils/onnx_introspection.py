from functools import cache
from typing import List

from inference_models.configuration import ONNXRUNTIME_EXECUTION_PROVIDERS


@cache
def get_selected_onnx_execution_providers() -> List[str]:
    try:
        import onnxruntime

        available_providers = set(onnxruntime.get_available_providers())
        return [
            ep for ep in ONNXRUNTIME_EXECUTION_PROVIDERS if ep in available_providers
        ]
    except ImportError:
        return []
