import os.path
from typing import List, Optional, Union

import torch

from inference.v1.configuration import DEFAULT_DEVICE
from inference.v1.weights_providers.core import get_model_from_provider

AnyModel = Union[str]


class AutoModel:

    def from_pretrained(
        self,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_trt_options: bool = True,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> AnyModel:
        if not os.path.isdir(model_name_or_path):
            model_metadata = get_model_from_provider(
                provider=weights_provider,
                model_id=model_name_or_path,
                api_key=api_key,
            )

        else:
            pass
