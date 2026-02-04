from typing import List, Union

import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.base.documents_parsing import TextOnlyOCRModel


class TROcrHF(TextOnlyOCRModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        **kwargs,
    ) -> "TextOnlyOCRModel":
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).to(device)
        processor = TrOCRProcessor.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        return cls(model=model, processor=processor, device=device)

    def __init__(
        self,
        processor: TrOCRProcessor,
        model: VisionEncoderDecoderModel,
        device: torch.device,
    ):
        self._processor = processor
        self._model = model
        self._device = device

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self._device)

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.inference_mode():
            return self._model.generate(pre_processed_images)

    def post_process(self, model_results: torch.Tensor, **kwargs) -> List[str]:
        decoded = self._processor.batch_decode(model_results, skip_special_tokens=True)
        return decoded
