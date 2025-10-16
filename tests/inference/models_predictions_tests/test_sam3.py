import numpy as np
import pytest

from inference.core.entities.requests.sam3 import Sam3SegmentationRequest, Sam3Prompt

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
        prompts=[Sam3Prompt(type="text", text="object")],
        output_prob_thresh=0.0,
    )
    resp = model.infer_from_request(req)
    assert resp.time >= 0
    assert hasattr(resp, "prompt_results")


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
        prompts=[
            Sam3Prompt(
                type="visual",
                boxes=[
                    Sam3Prompt.Box(
                        x=box[0] * img.shape[1],
                        y=box[1] * img.shape[0],
                        width=box[2] * img.shape[1],
                        height=box[3] * img.shape[0],
                    )
                ],
                box_labels=[1],
            )
        ],
        output_prob_thresh=0.0,
    )
    resp = model.infer_from_request(req)
    assert hasattr(resp, "prompt_results")


@pytest.mark.skipif(SegmentAnything3 is None, reason="SAM3 not installed")
def test_sam3_segment_with_rle_format(monkeypatch):
    import os

    # if os.getenv("SAM3_CHECKPOINT_PATH") is None:
    #     pytest.skip("SAM3_CHECKPOINT_PATH not set")

    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    model = SegmentAnything3(model_id="sam3")

    req = Sam3SegmentationRequest(
        image={"type": "numpy", "value": img},
        prompts=[Sam3Prompt(type="text", text="object")],
        format="rle",
        output_prob_thresh=0.0,
    )
    resp = model.infer_from_request(req)
    assert resp.time >= 0
    assert hasattr(resp, "prompt_results")
    if resp.prompt_results:
        first = resp.prompt_results[0]
        if first.predictions:
            pred = first.predictions[0]
            assert pred.format == "rle"
            assert isinstance(pred.masks, dict)
            assert "size" in pred.masks
            assert "counts" in pred.masks


@pytest.mark.skipif(SegmentAnything3 is None, reason="SAM3 not installed")
def test_sam3_segment_polygon_format_explicit(monkeypatch):
    import os

    # if os.getenv("SAM3_CHECKPOINT_PATH") is None:
    #     pytest.skip("SAM3_CHECKPOINT_PATH not set")

    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    model = SegmentAnything3(model_id="sam3")

    req = Sam3SegmentationRequest(
        image={"type": "numpy", "value": img},
        prompts=[Sam3Prompt(type="text", text="object")],
        format="polygon",
        output_prob_thresh=0.0,
    )
    resp = model.infer_from_request(req)
    assert resp.time >= 0
    assert hasattr(resp, "prompt_results")
    if resp.prompt_results:
        first = resp.prompt_results[0]
        if first.predictions:
            pred = first.predictions[0]
            assert pred.format == "polygon"
            assert isinstance(pred.masks, list)


@pytest.mark.skipif(SegmentAnything3 is None, reason="SAM3 not installed")
def test_sam3_segment_format_compatibility(monkeypatch):
    import os

    # if os.getenv("SAM3_CHECKPOINT_PATH") is None:
    #     pytest.skip("SAM3_CHECKPOINT_PATH not set")

    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    model = SegmentAnything3(model_id="sam3")

    # Without specifying format (should use default "polygon")
    req = Sam3SegmentationRequest(
        image={"type": "numpy", "value": img},
        prompts=[
            Sam3Prompt(
                type="visual",
                boxes=[
                    Sam3Prompt.Box(
                        x=0.25 * img.shape[1],
                        y=0.25 * img.shape[0],
                        width=0.5 * img.shape[1],
                        height=0.5 * img.shape[0],
                    )
                ],
                box_labels=[1],
            )
        ],
        output_prob_thresh=0.0,
    )
    resp = model.infer_from_request(req)
    assert hasattr(resp, "prompt_results")
    if resp.prompt_results and resp.prompt_results[0].predictions:
        assert resp.prompt_results[0].predictions[0].format == "polygon"
