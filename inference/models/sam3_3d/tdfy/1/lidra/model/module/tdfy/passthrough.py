import torch
from typing import Optional, Dict, Any, Tuple
import skimage.io as io
import os

from lidra.model.module.base import Base, TrainableBackbone
from lidra.data.utils import build_batch_extractor, empty_mapping
from lidra.data.dataset.return_type import extract_sample_uuid, extract_data


class Passthrough(Base):
    """
    A simple backbone that passes through the input.
    Useful for loading a pre-trained model and using it in callbacks.
    """

    def __init__(
        self,
        backbone: TrainableBackbone,
        batch_input_mapping: Dict[str, Any] = "input",
        batch_preprocessing_fn=None,
        **kwargs,
    ):
        models = {"backbone": backbone}
        self._batch_preprocessing_fn = (
            (lambda x: x) if batch_preprocessing_fn is None else batch_preprocessing_fn
        )
        super().__init__(models, **kwargs)
        self.batch_extractor_fn = build_batch_extractor(batch_input_mapping)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if batch is None:
            return None
        batch = self._batch_preprocessing_fn(batch)
        args, kwargs = self.batch_extractor_fn(batch)
        if len(args) > 0:
            raise ValueError(
                "Passthrough module expects a batch with no positional arguments"
            )
        return kwargs
