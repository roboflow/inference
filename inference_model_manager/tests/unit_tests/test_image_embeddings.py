from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.base.image_embeddings import ImageEmbeddingModel
from inference_models.models.clip.clip_onnx import ClipOnnx

from inference_model_manager.dispatch import invoke_action, list_actions, resolve_action
from inference_model_manager.image_embeddings import describe_image_embeddings
from inference_model_manager.model_manager import ModelManager
from inference_model_manager.registry_defaults import (
    IMAGE_EMBEDDINGS_ACTION,
    IMAGE_EMBEDDINGS_RESPONSE_TYPE,
    IMAGE_EMBEDDINGS_WITH_INFO_ACTION,
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


def test_describe_image_embeddings_prefers_the_model_embedding_preprocessing():
    class PretrainedEmbeddings(FakeClassifier):
        embedding_preprocessing = {"do_resize": True, "size": {"height": 224}}

    model = PretrainedEmbeddings()
    del model._inference_config

    result = describe_image_embeddings(model, torch.zeros((1, 4)), {})

    assert result["embedding_info"]["preprocessing"] == {
        "do_resize": True,
        "size": {"height": 224},
    }
    assert result["embedding_info"]["backend"] == "PretrainedEmbeddings"


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


def _weightless_clip(image_size=224):
    model = ClipOnnx(
        visual_onnx_session=MagicMock(),
        textual_onnx_session=MagicMock(),
        image_size=image_size,
        visual_input_name="image",
        textual_input_name="text",
        device=torch.device("cpu"),
        max_batch_size=8,
    )
    model.embed_images = MagicMock(
        return_value=torch.tensor([[0.25, 0.5, 1.0]], dtype=torch.float32)
    )
    return model


class FakeTextImageModel(TextImageEmbeddingModel):
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls()

    def embed_images(self, images, **kwargs):
        return torch.zeros((1, 2))

    def embed_text(self, texts, **kwargs):
        return torch.zeros((1, 2))


def test_clip_describes_its_image_embeddings_through_a_dedicated_action():
    model = _weightless_clip()

    result = invoke_action(
        model,
        action=IMAGE_EMBEDDINGS_WITH_INFO_ACTION,
        images=["image"],
        output_type="feature_vector",
        input_color_format="bgr",
        disable_preproc_grayscale=True,
    )

    assert torch.equal(result["embeddings"], torch.tensor([[0.25, 0.5, 1.0]]))
    assert result["embedding_info"] == {
        "feature_definition": "clip-image-embeds@v1",
        "output_type": "feature_vector",
        "normalization": "none",
        "dimension": 3,
        "preprocessing": model.embedding_preprocessing,
        "backend": "ClipOnnx",
        "precision": "torch.float32",
    }
    model.embed_images.assert_called_once_with(
        images=["image"], output_type="feature_vector", input_color_format="bgr"
    )


def test_clip_keeps_its_compact_default_image_embeddings():
    model = _weightless_clip()

    default_action, _ = resolve_action(model)
    _, entry = resolve_action(model, IMAGE_EMBEDDINGS_WITH_INFO_ACTION)
    result = invoke_action(model, action="embed_images", images=["image"])

    assert default_action == "embed_images"
    assert list_actions(model)["embed_images"]["response_type"] == (
        "roboflow-embeddings-compact-v1"
    )
    assert entry.method == "embed_images"
    assert entry.default is False
    assert entry.response_type == IMAGE_EMBEDDINGS_RESPONSE_TYPE
    assert isinstance(result, torch.Tensor)


def test_text_image_models_without_a_space_description_have_no_described_action():
    model = FakeTextImageModel()
    lazy_register(type(model))

    actions = list_actions(model)

    assert "embed_images" in actions
    assert IMAGE_EMBEDDINGS_WITH_INFO_ACTION not in actions
