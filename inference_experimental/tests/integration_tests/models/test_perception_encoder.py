import torch

from inference_exp.models.perception_encoder.perception_encoder import (
    PerceptionEncoder,
)


def test_perception_encoder_text_embedding():

    #TODO: this is a temporary path, should be replaced with a proper path / registry integration
    model = PerceptionEncoder.from_pretrained(
        "/tmp/cache/perception_encoder/PE-Core-B16-224" 
    )

    embeddings = model.embed_text("hello world")

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024) 