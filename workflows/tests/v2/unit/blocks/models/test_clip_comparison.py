"""``roboflow_core/clip_comparison@v2`` called directly with a fake embedding model.

The fake's red, green and blue labels are unit color axes and an image's
embedding is its mean color, so similarities are exact: a pure red image
scores [0, 1, 0] against ["green", "red", "blue"]. Not real inference.
"""

from typing import Any, Dict, List

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.clip_comparison.v2 import DEFAULT_MODEL_ID, ClipComparisonBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData

from tests.v2.support.models.fake_embedding_models import (
    FakeEmbeddingModel,
    color_image,
)
from tests.v2.support.models.fake_task_models import FakeLoader

COLORS = ["green", "red", "blue"]


def _image(red: int, green: int, blue: int, image_id: str = "a") -> ImageData:
    image = ImageData.from_tensor(color_image(red, green, blue), image_id=image_id)

    return image


def _run(model: Any, images: List[ImageData], **params: Any) -> List[Dict[str, Any]]:
    block = ClipComparisonBlock(models={DEFAULT_MODEL_ID: model})
    params.setdefault("classes", COLORS)
    results = block.run(images=Batch(images), **params)

    return results


def test_red_image_is_most_similar_to_red() -> None:
    (result,) = _run(FakeEmbeddingModel(), [_image(255, 0, 0)])

    assert torch.equal(result["similarities"], torch.tensor([0.0, 1.0, 0.0]))
    assert (result["most_similar_class"], result["max_similarity"]) == ("red", 1.0)
    assert (result["least_similar_class"], result["min_similarity"]) == ("green", 0.0)


def test_prediction_is_a_one_image_classification_of_similarities() -> None:
    (result,) = _run(FakeEmbeddingModel(), [_image(0, 0, 255, "frame-7")])
    prediction = result["classification_predictions"]

    assert prediction.class_id.tolist() == [2]
    assert torch.equal(prediction.confidence, result["similarities"].unsqueeze(0))
    metadata = prediction.images_metadata[0]
    assert metadata["class_names"] == {0: "green", 1: "red", 2: "blue"}
    assert metadata["prediction_type"] == "classification"
    assert frame_of(prediction).image_id == "frame-7"


def test_ties_pick_the_first_label_for_both_ends() -> None:
    (result,) = _run(FakeEmbeddingModel(), [_image(90, 90, 90)])

    assert result["most_similar_class"] == result["least_similar_class"] == "green"
    assert result["max_similarity"] == pytest.approx(result["min_similarity"])


def test_black_image_scores_zero_instead_of_nan() -> None:
    (result,) = _run(FakeEmbeddingModel(), [_image(0, 0, 0)])

    assert torch.equal(result["similarities"], torch.zeros(3))


def test_fixed_labels_embed_once_across_runs_and_images_batch() -> None:
    model = FakeEmbeddingModel()
    block = ClipComparisonBlock(models={DEFAULT_MODEL_ID: model})
    images = Batch([_image(255, 0, 0, "a"), _image(0, 255, 0, "b")])

    first = block.run(images=images, classes=COLORS)
    second = block.run(images=images, classes=COLORS)

    assert model.text_calls == [COLORS]
    assert model.image_calls == [2, 2]
    assert [row["most_similar_class"] for row in first + second] == ["red", "green"] * 2


def test_labels_may_change_between_runs_and_only_new_labels_are_embedded() -> None:
    model = FakeEmbeddingModel()
    block = ClipComparisonBlock(models={DEFAULT_MODEL_ID: model})
    images = Batch([_image(255, 0, 0, "a"), _image(0, 0, 255, "b")])

    first = block.run(images=images, classes=["red", "blue"])
    second = block.run(images=images, classes=["green", "blue"])

    assert model.image_calls == [2, 2]
    assert model.text_calls == [["red", "blue"], ["green"]]
    assert [row["most_similar_class"] for row in first] == ["red", "blue"]
    assert [row["most_similar_class"] for row in second] == ["green", "blue"]
    assert second[1]["similarities"].tolist() == [0.0, 1.0]


def test_per_image_labels_fail_before_any_model_call() -> None:
    model = FakeEmbeddingModel()
    classes = Batch([["red", "blue"], ["green", "blue"]])

    with pytest.raises(ValueError, match="classes is one value for every image"):
        _run(model, [_image(255, 0, 0, "a"), _image(0, 0, 255, "b")], classes=classes)

    assert model.image_calls == model.text_calls == []


def test_the_model_may_change_between_runs_and_loads_once() -> None:
    loader = FakeLoader(FakeEmbeddingModel)
    block = ClipComparisonBlock(models=ModelCache(loader))
    images = Batch(
        [_image(255, 0, 0, "a"), _image(0, 255, 0, "b"), _image(0, 0, 255, "c")]
    )

    for model_id in ("x", "y", "x"):
        results = block.run(images=images, classes=COLORS, model_id=model_id)

    assert loader.loaded == [ModelRequest("x"), ModelRequest("y")]
    assert [model.image_calls for model in loader.models] == [[3, 3], [3]]
    assert [row["most_similar_class"] for row in results] == ["red", "green", "blue"]


def test_per_image_model_ids_fail_before_any_load() -> None:
    loader = FakeLoader(FakeEmbeddingModel)
    block = ClipComparisonBlock(models=ModelCache(loader))

    with pytest.raises(ValueError, match="model_id is one value for every image"):
        block.run(
            images=Batch([_image(255, 0, 0, "a"), _image(0, 255, 0, "b")]),
            classes=COLORS,
            model_id=Batch(["x", "y"]),
        )

    assert loader.loaded == []


def test_similarities_keep_the_model_dtype() -> None:
    model = FakeEmbeddingModel(dtype=torch.float64)

    (result,) = _run(model, [_image(255, 0, 0)])

    assert result["similarities"].dtype == torch.float64
    assert result["classification_predictions"].confidence.dtype == torch.float64


def test_literal_label_list_must_not_be_empty() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        ClipComparisonBlock.Params(images="$inputs.image", classes=[])


def test_default_model_is_clip_vit_b_16() -> None:
    params = ClipComparisonBlock.Params(images="$inputs.image", classes=COLORS)

    (resource,) = ClipComparisonBlock.discover_dependent_resources(params)

    assert params.model_id == "clip/ViT-B-16"
    assert resource.identifier == "clip/ViT-B-16"


def test_a_model_without_embedding_methods_fails_with_its_name() -> None:
    with pytest.raises(TypeError, match="has no embed_images or embed_text"):
        _run(object(), [_image(1, 2, 3)])


def test_a_non_string_label_fails_before_any_model_call() -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(ValueError, match=r"must be text labels, got \[3\]"):
        _run(model, [_image(1, 2, 3)], classes=["red", 3])
    assert model.image_calls == [] and model.text_calls == []
