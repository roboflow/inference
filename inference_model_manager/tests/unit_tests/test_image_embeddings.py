from types import SimpleNamespace

import numpy as np
import pytest
import torch
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.image_embeddings import ImageEmbeddingModel

from inference_model_manager.dispatch import invoke_action, list_actions
from inference_model_manager.image_embeddings import describe_image_embeddings
from inference_model_manager.model_manager import ModelManager
from inference_model_manager.registry_defaults import (
    IMAGE_EMBEDDINGS_ACTION,
    IMAGE_EMBEDDINGS_RESPONSE_TYPE,
    lazy_register,
)
from inference_model_manager.serializers_typed import serialize_image_embeddings


class _Section:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self, mode: str) -> dict:
        assert mode == "json"
        return dict(self._payload)


class FakeClassifier(ImageEmbeddingModel):
    def __init__(self) -> None:
        self._inference_config = SimpleNamespace(
            image_pre_processing=_Section({"auto-orient": {"enabled": True}}),
            network_input=_Section({"training_input_size": {"width": 224}}),
        )
        self.calls = []

    def infer(self, images, **kwargs):
        return []

    def embed_images(self, images, output_type=None, **kwargs):
        self.calls.append((images, output_type, kwargs))
        return torch.tensor([[0.5, -1.5]], dtype=torch.float32)

    def get_embedding_info(self, output_type=None):
        output_type = output_type or "feature_vector"
        definition = (
            "classifier-linear-output@v1"
            if output_type == "logits"
            else "classifier-linear-input@v1"
        )
        return {
            "feature_definition": definition,
            "output_type": output_type,
            "normalization": "none",
        }


def test_embed_images_action_is_registered_for_classifier_embedding_models():
    model = FakeClassifier()
    lazy_register(type(model))

    actions = list_actions(model)

    assert actions[IMAGE_EMBEDDINGS_ACTION]["default"] is False
    assert actions[IMAGE_EMBEDDINGS_ACTION]["response_type"] == (
        IMAGE_EMBEDDINGS_RESPONSE_TYPE
    )
    assert {
        "images",
        "output_type",
        "input_color_format",
        "disable_preproc_contrast",
        "disable_preproc_grayscale",
        "disable_preproc_static_crop",
    } <= set(actions[IMAGE_EMBEDDINGS_ACTION]["params"])


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_invoke_action_pairs_embeddings_with_space_metadata(output_type):
    model = FakeClassifier()
    lazy_register(type(model))

    result = invoke_action(
        model,
        action=IMAGE_EMBEDDINGS_ACTION,
        images=["image"],
        output_type=output_type,
        input_color_format="rgb",
        disable_preproc_grayscale=True,
    )

    assert torch.equal(result["embeddings"], torch.tensor([[0.5, -1.5]]))
    assert result["embedding_info"] == {
        "feature_definition": (
            "classifier-linear-output@v1"
            if output_type == "logits"
            else "classifier-linear-input@v1"
        ),
        "output_type": output_type,
        "normalization": "none",
        "dimension": 2,
        "preprocessing": {
            "image_pre_processing": {"auto-orient": {"enabled": True}},
            "network_input": {"training_input_size": {"width": 224}},
        },
        "backend": "FakeClassifier",
        "precision": "torch.float32",
    }
    assert model.calls == [
        (
            ["image"],
            output_type,
            {
                "input_color_format": "rgb",
                "pre_processing_overrides": PreProcessingOverrides(
                    disable_grayscale=True
                ),
            },
        )
    ]


def test_describe_image_embeddings_reports_only_model_side_preprocessing():
    model = FakeClassifier()

    result = describe_image_embeddings(
        model, torch.zeros((2, 3)), {"images": ["a", "b"]}
    )

    assert result["embedding_info"]["dimension"] == 3
    assert result["embedding_info"]["output_type"] == "feature_vector"
    assert list(result["embedding_info"]["preprocessing"]) == [
        "image_pre_processing",
        "network_input",
    ]


def test_wire_marshalling_turns_the_embedding_tensor_into_numpy():
    model = FakeClassifier()
    lazy_register(type(model))
    raw = invoke_action(model, action=IMAGE_EMBEDDINGS_ACTION, images=["image"])

    marshalled = ModelManager._wire_marshal_result(raw, 1)

    assert isinstance(marshalled["embeddings"], np.ndarray)
    assert marshalled["embeddings"].tolist() == [[0.5, -1.5]]
    assert marshalled["embedding_info"]["precision"] == "torch.float32"


def test_serialize_image_embeddings_keeps_the_envelope():
    info = {"feature_definition": "classifier-linear-input@v1", "dimension": 2}

    serialized = serialize_image_embeddings(
        {"embeddings": np.array([[0.5, -1.5]]), "embedding_info": info}, model=None
    )

    assert serialized["type"] == IMAGE_EMBEDDINGS_RESPONSE_TYPE
    assert serialized["embeddings"].tolist() == [[0.5, -1.5]]
    assert serialized["embedding_info"] is info
