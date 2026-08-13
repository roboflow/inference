import numpy as np
import torch

from inference_models.models.owlv2.entities import (
    ReferenceBoundingBox,
    ReferenceExample,
    ReferenceExamplesEmbeddings,
)
from inference_models.models.owlv2.owlv2_hf import OWLv2HF

IMAGE_URL = "https://storage.googleapis.com/bucket/workspace/image-1/original.jpg"


def build_model_shell() -> OWLv2HF:
    return OWLv2HF(
        model=None,
        processor=None,
        device=torch.device("cpu"),
        owlv2_class_embeddings_cache=None,
        owlv2_images_embeddings_cache=None,
        allow_url_input=True,
        allow_non_https_url=False,
        allow_url_without_fqdn=False,
        whitelisted_domains=None,
        blacklisted_domains=None,
        allow_local_storage_access_for_reference_images=False,
    )


def stub_embedding_internals(model: OWLv2HF, monkeypatch, captured: dict) -> None:
    def fake_prepare(reference_examples, iou_threshold):
        captured["reference_examples"] = reference_examples
        return ReferenceExamplesEmbeddings(class_embeddings={}, image_embeddings=None)

    monkeypatch.setattr(model, "prepare_reference_examples_embeddings", fake_prepare)
    monkeypatch.setattr(
        model, "infer_with_reference_examples_embeddings", lambda **kwargs: []
    )


def test_infer_with_reference_examples_coerces_wire_dicts(monkeypatch) -> None:
    model = build_model_shell()
    captured: dict = {}
    stub_embedding_internals(model, monkeypatch, captured)

    result = model.infer_with_reference_examples(
        images=np.zeros((16, 16, 3), dtype=np.uint8),
        reference_examples=[
            {
                "image": IMAGE_URL,
                "boxes": [
                    {"x": 10, "y": 20, "w": 30, "h": 40, "cls": "screw"},
                    {"x": 1, "y": 2, "w": 3, "h": 4, "cls": "bolt", "negative": True},
                ],
            }
        ],
    )

    assert result == []
    coerced = captured["reference_examples"]
    assert len(coerced) == 1
    assert isinstance(coerced[0], ReferenceExample)
    assert coerced[0].image == IMAGE_URL
    assert [box.cls for box in coerced[0].boxes] == ["screw", "bolt"]
    assert coerced[0].boxes[0].negative is False
    assert coerced[0].boxes[1].negative is True


def test_infer_with_reference_examples_passes_through_typed_examples(
    monkeypatch,
) -> None:
    model = build_model_shell()
    captured: dict = {}
    stub_embedding_internals(model, monkeypatch, captured)
    typed_example = ReferenceExample(
        image=IMAGE_URL,
        boxes=[ReferenceBoundingBox(x=10, y=20, w=30, h=40, cls="screw")],
    )

    result = model.infer_with_reference_examples(
        images=np.zeros((16, 16, 3), dtype=np.uint8),
        reference_examples=[typed_example],
    )

    assert result == []
    assert captured["reference_examples"][0] is typed_example
