"""``v2/clip@v1`` and ``v2/perception_encoder@v1`` called directly with fakes.

The fake (``tests/v2/support/models/fake_embedding_models.py``) tests the
model-resource contract, batching, the text cache and errors; it is not real
inference. Real packages are covered, when present, by
``integration/external/models/test_embedding_models.py``.
"""

from typing import Any, Dict

import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    ModelRequestConflict,
)
from roboflow_workflows.v2.blocks.models.embeddings import (
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_PERCEPTION_ENCODER_MODEL_ID,
    ClipEmbeddingBlock,
    PerceptionEncoderEmbeddingBlock,
    TextEmbeddingCache,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData

from tests.v2.support.models.fake_embedding_models import (
    FakeEmbeddingModel,
    color_image,
    text_embedding,
)
from tests.v2.support.models.fake_task_models import FakeLoader


def _image(red: int, green: int, blue: int, image_id: str, **size: int) -> ImageData:
    image = ImageData.from_tensor(
        color_image(red, green, blue, **size), image_id=image_id
    )

    return image


def _block(models: Dict[Any, Any], block_type: type = ClipEmbeddingBlock):
    block = block_type(models=models)

    return block


# --- embeddings ---------------------------------------------------------------


def test_literal_text_gives_one_embedding_of_the_default_model() -> None:
    model = FakeEmbeddingModel()
    block = _block({DEFAULT_CLIP_MODEL_ID: model})

    result = block.run(data="red", model_id=DEFAULT_CLIP_MODEL_ID)

    assert set(result) == {"embedding"}
    assert torch.equal(result["embedding"], text_embedding("red"))
    assert result["embedding"].shape == (4,)


def test_images_of_different_sizes_embed_in_one_call_in_row_order() -> None:
    model = FakeEmbeddingModel()
    block = _block({"m": model})
    images = Batch(
        [
            _image(255, 0, 0, "a"),
            _image(0, 255, 0, "b", height=5),
            _image(0, 0, 255, "c", width=3),
        ]
    )

    results = block.run(data=images, model_id="m")

    assert model.image_calls == [3]
    assert model.color_formats == ["rgb"]
    embeddings = torch.stack([result["embedding"] for result in results])
    assert torch.equal(embeddings[:, :3], torch.eye(3))


def test_empty_batch_runs_no_model() -> None:
    model = FakeEmbeddingModel()

    results = _block({"m": model}).run(data=Batch([]), model_id="m")

    assert results == []
    assert model.image_calls == [] and model.text_calls == []


def test_per_row_model_ids_load_each_model_once_and_group_rows() -> None:
    loader = FakeLoader(FakeEmbeddingModel)
    block = _block(ModelCache(loader))
    images = Batch(
        [_image(255, 0, 0, "a"), _image(0, 255, 0, "b"), _image(9, 9, 9, "c")]
    )

    results = block.run(data=images, model_id=Batch(["x", "y", "x"]))

    assert loader.loaded == [ModelRequest("x"), ModelRequest("y")]
    first, second = loader.models
    assert (first.image_calls, second.image_calls) == ([2], [1])
    assert torch.equal(results[1]["embedding"][:3], torch.tensor([0.0, 1.0, 0.0]))


def test_mixed_image_and_text_rows_of_one_model_keep_row_order() -> None:
    model = FakeEmbeddingModel()
    data = Batch(["blue", _image(255, 0, 0, "a"), "green", _image(0, 0, 255, "b")])

    results = _block({"m": model}).run(data=data, model_id="m")

    assert model.image_calls == [2]
    assert model.text_calls == [["blue", "green"]]
    assert torch.equal(results[0]["embedding"], text_embedding("blue"))
    assert torch.equal(results[1]["embedding"][:3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(results[2]["embedding"], text_embedding("green"))
    assert torch.equal(results[3]["embedding"][:3], torch.tensor([0.0, 0.0, 1.0]))


def test_output_keeps_the_model_device_and_dtype() -> None:
    model = FakeEmbeddingModel(device="meta", dtype=torch.float64)
    block = _block({"m": model})

    image_result = block.run(data=Batch([_image(1, 2, 3, "a")]), model_id="m")[0]
    text_result = block.run(data="red", model_id="m")

    for embedding in (image_result["embedding"], text_result["embedding"]):
        assert (embedding.device.type, embedding.dtype) == ("meta", torch.float64)


def test_perception_encoder_block_has_its_own_default_model() -> None:
    model = FakeEmbeddingModel()
    block = _block(
        {DEFAULT_PERCEPTION_ENCODER_MODEL_ID: model}, PerceptionEncoderEmbeddingBlock
    )
    params = PerceptionEncoderEmbeddingBlock.Params(data="$inputs.image")

    result = block.run(data="blue", model_id=params.model_id)

    assert params.model_id == "perception-encoder/PE-Core-L14-336"
    assert torch.equal(result["embedding"], text_embedding("blue"))


def test_dependency_names_the_selected_model_and_package() -> None:
    params = ClipEmbeddingBlock.Params(
        data="$inputs.image", model_package_id="$inputs.package"
    )

    (resource,) = ClipEmbeddingBlock.discover_dependent_resources(params)

    assert resource.identifier == DEFAULT_CLIP_MODEL_ID
    assert resource.details == {
        "resource": "models",
        "task_type": "embedding",
        "model_package_id": "$inputs.package",
    }


# --- text cache ----------------------------------------------------------------


def test_fixed_texts_run_the_text_encoder_once() -> None:
    model = FakeEmbeddingModel()
    block = _block({"m": model})

    first = block.run(data=Batch(["red", "blue", "red"]), model_id="m")
    second = block.run(data=Batch(["red", "blue", "red"]), model_id="m")

    assert model.text_calls == [["red", "blue"]]
    for a, b in zip(first, second):
        assert torch.equal(a["embedding"], b["embedding"])


def test_new_texts_embed_only_what_is_missing() -> None:
    model = FakeEmbeddingModel()
    block = _block({"m": model})

    block.run(data=Batch(["red", "blue"]), model_id="m")
    results = block.run(data=Batch(["green", "red", "a ship"]), model_id="m")

    assert model.text_calls == [["red", "blue"], ["green", "a ship"]]
    assert torch.equal(results[2]["embedding"], text_embedding("a ship"))


def test_cache_key_includes_the_model_request() -> None:
    loader = FakeLoader(FakeEmbeddingModel)
    block = _block(ModelCache(loader))

    block.run(data="red", model_id="x")
    block.run(data="red", model_id="x", model_package_id="p")
    block.run(data="red", model_id="y")

    assert [model.text_calls for model in loader.models] == [[["red"]]] * 3


def test_returned_embeddings_are_not_the_cached_ones() -> None:
    model = FakeEmbeddingModel()
    block = _block({"m": model})

    block.run(data="red", model_id="m")["embedding"].zero_()
    again = block.run(data="red", model_id="m")["embedding"]

    assert torch.equal(again, text_embedding("red"))
    assert model.text_calls == [["red"]]


def test_cache_is_bounded_and_drops_the_least_recently_used_text() -> None:
    model = FakeEmbeddingModel()
    cache = TextEmbeddingCache(capacity=2)
    request = ModelRequest("m")

    cache.embed(["a", "b"], model=model, request=request)
    cache.embed(["a"], model=model, request=request)  # "b" is now the oldest
    cache.embed(["c"], model=model, request=request)
    cache.embed(["a", "b"], model=model, request=request)

    assert len(cache) == 2
    assert model.text_calls == [["a", "b"], ["c"], ["b"]]


def test_zero_capacity_cache_embeds_every_call() -> None:
    model = FakeEmbeddingModel()
    cache = TextEmbeddingCache(capacity=0)

    for _ in range(2):
        cache.embed(["a"], model=model, request=ModelRequest("m"))

    assert model.text_calls == [["a"], ["a"]]
    assert len(cache) == 0


# --- supplied models and errors ------------------------------------------------


def test_supplied_model_rejects_a_package_it_was_not_supplied_for() -> None:
    block = _block({DEFAULT_CLIP_MODEL_ID: FakeEmbeddingModel()})

    with pytest.raises(ModelRequestConflict, match="package is unverified"):
        block.run(data="red", model_id=DEFAULT_CLIP_MODEL_ID, model_package_id="onnx")


def test_supplied_model_serves_its_exact_package_request() -> None:
    model = FakeEmbeddingModel()
    block = _block({ModelRequest("m", "onnx"): model})

    block.run(data="red", model_id="m", model_package_id="onnx")

    assert model.text_calls == [["red"]]


def test_a_model_without_embedding_methods_fails_with_its_name() -> None:
    class Detector:
        def pre_process(self, images: Any) -> None:
            pass

    with pytest.raises(
        TypeError, match=r"'m' \(Detector\) has no embed_images or embed_text"
    ):
        _block({"m": Detector()}).run(data="red", model_id="m")


def test_a_non_rgb_image_fails_before_any_model_call() -> None:
    model = FakeEmbeddingModel()
    gray = ImageData.from_tensor(
        torch.zeros((1, 4, 4), dtype=torch.uint8), image_id="g"
    )

    with pytest.raises(ValueError, match="3-channel RGB"):
        _block({"m": model}).run(data=Batch([_image(1, 1, 1, "a"), gray]), model_id="m")
    assert model.image_calls == []


def test_a_wrong_number_of_model_embeddings_fails() -> None:
    class Short(FakeEmbeddingModel):
        def embed_images(self, images: Any, **kwargs: Any) -> torch.Tensor:
            return super().embed_images(images, **kwargs)[:1]

    images = Batch([_image(1, 1, 1, "a"), _image(2, 2, 2, "b")])

    with pytest.raises(
        ValueError, match=r"returned \(1, 4\) torch.float32 for 2 inputs"
    ):
        _block({"m": Short()}).run(data=images, model_id="m")


def test_a_per_row_model_id_batch_of_another_length_fails() -> None:
    block = _block({"m": FakeEmbeddingModel()})

    with pytest.raises(ValueError, match="model_id has 1 values for 2 rows"):
        block.run(data=Batch(["red", "blue"]), model_id=Batch(["m"]))


def test_negative_cache_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 0"):
        TextEmbeddingCache(capacity=-1)
