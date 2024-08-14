import numpy as np
import pytest
import torch

from inference.models.sam2 import SegmentAnything2
from inference.core.entities.requests.sam2 import Sam2PromptSet





@pytest.mark.slow
def test_sam2_single_image_embedding(
    sam2_small_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    # when
    embedding, img_shape, id_ = model.embed_image(example_image)

    # then
    assert torch.sum(embedding['image_embed']) == 25155.9336, "embedding sum doesnt drift/change"
    assert img_shape == (427,640) , "Image shape must match the expected shape" 
    


