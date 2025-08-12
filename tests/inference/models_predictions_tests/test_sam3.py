import numpy as np
import pytest

from inference.core.entities.requests.sam3 import Sam3SegmentationRequest

try:
    from inference.models.sam3 import SegmentAnything3
except ModuleNotFoundError:
    # SAM3 optional dependency
    SegmentAnything3 = None


pytestmark = pytest.mark.slow


@pytest.mark.skipif(SegmentAnything3 is None, reason="SAM3 not installed")
def test_sam3_embed_and_segment_with_text(monkeypatch):
    # Set required env for SAM3 to run in CI-like environment (user must provide real checkpoint to actually run)
    import os

    # if os.getenv("SAM3_CHECKPOINT_PATH") is None:
    #     pytest.skip("SAM3_CHECKPOINT_PATH not set")

    # Create a synthetic image
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)

    model = SegmentAnything3(model_id="sam3")
    emb = model.embed_image(img)
    assert emb.image_id is not None

    req = Sam3SegmentationRequest(
        image={"type": "numpy", "value": img},
        text="object",
        output_prob_thresh=0.0,
    )
    resp = model.infer_from_request(req)
    assert resp.time >= 0
    # Predictions may be empty depending on checkpoint/text; just ensure well-formed structure
    assert hasattr(resp, "predictions")


@pytest.mark.skipif(SegmentAnything3 is None, reason="SAM3 not installed")
def test_sam3_segment_with_box_prompt(monkeypatch):
    import os

    # if os.getenv("SAM3_CHECKPOINT_PATH") is None:
    #     pytest.skip("SAM3_CHECKPOINT_PATH not set")

    img = (np.random.rand(160, 200, 3) * 255).astype(np.uint8)
    model = SegmentAnything3(model_id="sam3")

    box = [0.25, 0.25, 0.5, 0.5]
    req = Sam3SegmentationRequest(
        image={"type": "numpy", "value": img},
        boxes=[box],
        box_labels=[1],
        output_prob_thresh=0.0,
    )
    resp = model.infer_from_request(req)
    assert hasattr(resp, "predictions")


