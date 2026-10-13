"""Embedding blocks in compiled V2 workflows with a ``ModelCache``.

A fake embedding model (``tests/v2/support/models/fake_embedding_models.py``)
stands in for CLIP behind the ``models`` resource; the engine, compiler and
the other blocks are real. Nothing is downloaded and this is not real
inference. The tests check what a workflow builder relies on:

* CLIP image embeddings feed ``v2/identify_changes@v1``, which flags the frame
  where the scene changes;
* an image and a text embedding meet in ``v2/cosine_similarity@v1``;
* ``v2/clip_comparison@v1`` labels each image, with labels from an input;
* models load lazily once per request, fixed texts run the text encoder once,
  and a gated image is skipped.
"""

from typing import Any, Dict, List

import pytest
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.clip_comparison import ClipComparisonBlock
from roboflow_workflows.v2.blocks.models.embeddings import (
    ClipEmbeddingBlock,
    PerceptionEncoderEmbeddingBlock,
)
from roboflow_workflows.v2.blocks.sampling import IdentifyChangesBlock
from roboflow_workflows.v2.blocks.transforms.similarity import CosineSimilarityBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.compilation.fixture_blocks import Gate
from tests.v2.support.models.fake_embedding_models import (
    FakeEmbeddingModel,
    color_image,
)
from tests.v2.support.models.fake_task_models import FakeLoader

MODES = ("run", "phases")
CLIP = "clip/ViT-B-32"
CATALOGUE = Catalogue(
    [
        ClipEmbeddingBlock,
        PerceptionEncoderEmbeddingBlock,
        ClipComparisonBlock,
        CosineSimilarityBlock,
        IdentifyChangesBlock,
        Gate,
    ],
    kinds=NATIVE_KINDS,
)
IMAGES = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}


def _image(red: int, green: int, blue: int, image_id: str) -> ImageData:
    image = ImageData.from_tensor(color_image(red, green, blue), image_id=image_id)

    return image


def _outputs(**selectors: str) -> List[Dict[str, str]]:
    outputs = [
        {"type": "JsonField", "name": name, "selector": selector}
        for name, selector in selectors.items()
    ]

    return outputs


def _session(definition: Dict[str, Any], models: Any, *, mode: str = "run"):
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session({"models": models})

    return session


# --- CLIP -> identify_changes ---------------------------------------------------


def _scene_frames() -> List[ImageData]:
    # Twelve reddish frames that wobble slightly, then one blue frame.
    frames = [
        _image(250 - 5 * (i % 3), 10 + 3 * (i % 4), 5, f"f{i}") for i in range(12)
    ]
    frames.append(_image(5, 10, 250, "f12"))

    return frames


@pytest.mark.parametrize("mode", MODES)
def test_clip_embeddings_let_identify_changes_flag_the_scene_change(mode: str) -> None:
    definition = {
        "version": "2.0",
        "inputs": [IMAGES],
        "steps": [
            {"type": "v2/clip@v1", "name": "clip", "data": "$inputs.image"},
            {
                "type": "v2/identify_changes@v1",
                "name": "changes",
                "embedding": "$steps.clip.embedding",
                "threshold_percentile": 0.02,
                "warmup": 4,
            },
        ],
        "outputs": _outputs(
            changed="$steps.changes.is_outlier",
            warming_up="$steps.changes.warming_up",
        ),
    }
    loader = FakeLoader(FakeEmbeddingModel)
    session = _session(definition, ModelCache(loader), mode=mode)

    rows = [session.run({"image": [frame]}).rows()[0] for frame in _scene_frames()]

    assert [n for n, row in enumerate(rows) if row["changed"]] == [12]
    assert [row["warming_up"] for row in rows[:5]] == [True] * 4 + [False]
    assert loader.loaded == [ModelRequest(CLIP)]
    assert loader.models[0].image_calls == [1] * 13


# --- image and text embeddings -> cosine similarity -----------------------------


def _image_to_text_definition() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [IMAGES],
        "steps": [
            {"type": "v2/clip@v1", "name": "image_embedding", "data": "$inputs.image"},
            {"type": "v2/clip@v1", "name": "text_embedding", "data": "red"},
            {
                "type": "v2/cosine_similarity@v1",
                "name": "similarity",
                "embedding_1": "$steps.image_embedding.embedding",
                "embedding_2": "$steps.text_embedding.embedding",
            },
        ],
        "outputs": _outputs(similarity="$steps.similarity.similarity"),
    }

    return definition


def test_image_and_literal_text_embeddings_compare_per_image() -> None:
    model = FakeEmbeddingModel()
    session = _session(_image_to_text_definition(), {CLIP: model})
    images = [_image(255, 0, 0, "red"), _image(0, 255, 0, "green")]

    first = session.run({"image": images}).rows()
    second = session.run({"image": images}).rows()

    assert [row["similarity"] for row in first + second] == pytest.approx(
        [1.0, 0.0, 1.0, 0.0]
    )
    assert model.text_calls == [["red"]]
    assert model.image_calls == [2, 2]


def test_a_mapping_without_the_model_fails_when_the_step_runs() -> None:
    session = _session(_image_to_text_definition(), {"another": FakeEmbeddingModel()})

    with pytest.raises(
        StepExecutionError, match="'clip/ViT-B-32' is not in the 'models'"
    ):
        session.run({"image": [_image(1, 2, 3, "a")]})


# --- clip comparison --------------------------------------------------------------


def _comparison_definition(**step: Any) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [
            IMAGES,
            {"type": "WorkflowParameter", "name": "classes"},
            {"type": "WorkflowParameter", "name": "package", "default_value": None},
            {"type": "WorkflowBatchInput", "name": "keep"},
        ],
        "steps": [
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.compare"],
            },
            {
                "type": "v2/clip_comparison@v1",
                "name": "compare",
                "images": "$inputs.image",
                "classes": "$inputs.classes",
                "model_package_id": "$inputs.package",
                **step,
            },
        ],
        "outputs": _outputs(
            label="$steps.compare.most_similar_class",
            score="$steps.compare.max_similarity",
            prediction="$steps.compare.classification_predictions",
        ),
    }

    return definition


@pytest.mark.parametrize("mode", MODES)
def test_comparison_labels_images_and_skips_a_gated_one(mode: str) -> None:
    loader = FakeLoader(FakeEmbeddingModel)
    session = _session(_comparison_definition(), ModelCache(loader), mode=mode)
    images = [_image(0, 0, 255, "a"), _image(255, 0, 0, "b"), _image(0, 255, 0, "c")]
    inputs = {"image": images, "classes": ["red", "green", "blue"]}

    rows = session.run({**inputs, "keep": [True, False, True]}).rows()

    assert [row["label"] for row in rows] == ["blue", None, "green"]
    assert rows[0]["score"] == pytest.approx(1.0)
    assert rows[2]["prediction"].class_id.tolist() == [1]
    assert loader.loaded == [ModelRequest("clip/ViT-B-16")]
    assert loader.models[0].image_calls == [2]


def test_comparison_selects_the_model_package_from_an_input() -> None:
    loader = FakeLoader(FakeEmbeddingModel)
    session = _session(_comparison_definition(), ModelCache(loader))
    inputs = {"image": [_image(255, 0, 0, "a")], "classes": ["red"], "keep": [True]}

    session.run({**inputs, "package": "onnx-fp16"})
    session.run(inputs)

    assert loader.loaded == [
        ModelRequest("clip/ViT-B-16", "onnx-fp16"),
        ModelRequest("clip/ViT-B-16"),
    ]


def test_comparison_labels_that_change_between_runs_stay_correct() -> None:
    model = FakeEmbeddingModel()
    session = _session(_comparison_definition(), {"clip/ViT-B-16": model})
    image = {"image": [_image(255, 0, 0, "a")], "keep": [True]}

    first = session.run({**image, "classes": ["green", "red"]}).rows()[0]
    second = session.run({**image, "classes": ["blue", "green"]}).rows()[0]
    third = session.run({**image, "classes": ["green", "red"]}).rows()[0]

    assert (first["label"], second["label"], third["label"]) == ("red", "blue", "red")
    assert second["score"] == pytest.approx(0.0)
    assert model.text_calls == [["green", "red"], ["blue"]]


@pytest.mark.parametrize(
    "classes, reason",
    [([], "classes is invalid.*at least 1 item"), (["red", 3], "must be text labels")],
    ids=["empty", "number"],
)
def test_comparison_rejects_invalid_selected_labels_when_it_runs(
    classes: List[Any], reason: str
) -> None:
    model = FakeEmbeddingModel()
    session = _session(_comparison_definition(), {"clip/ViT-B-16": model})

    with pytest.raises(StepExecutionError, match=reason):
        session.run(
            {"image": [_image(1, 2, 3, "a")], "classes": classes, "keep": [True]}
        )
    assert model.image_calls == [] and model.text_calls == []


def test_comparison_rejects_an_empty_per_image_label_list_when_it_runs() -> None:
    model = FakeEmbeddingModel()
    definition = _comparison_definition()
    definition["inputs"][1] = {"type": "WorkflowBatchInput", "name": "classes"}
    session = _session(definition, {"clip/ViT-B-16": model})
    images = [_image(1, 2, 3, "a"), _image(4, 5, 6, "b")]

    with pytest.raises(
        StepExecutionError, match=r"index \[1\].*classes is invalid.*at least 1 item"
    ):
        session.run({"image": images, "classes": [["red"], []], "keep": [True, True]})
    assert model.image_calls == []


def test_comparison_rejects_a_literal_empty_label_list_at_compile_time() -> None:
    with pytest.raises(ParamsValidationError, match="classes"):
        compile_workflow(_comparison_definition(classes=[]), catalogue=CATALOGUE)
