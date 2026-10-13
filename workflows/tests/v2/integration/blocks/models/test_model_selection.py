"""Model and package selection of ``v2/object_detection@v1`` through a cache.

Compiled V2 workflows run with a ``ModelCache`` whose loader builds fake
models with the native ``inference_models`` phase API. Each fake model
stamps its request into the predictions, so a test sees which model served
which image. This checks selection, grouping and reuse; it is not real model
inference. A per-image ``model_package_id`` of ``None`` asks the loader to
choose for that image; a gated image is skipped and gets no prediction.
"""

from typing import Any, Dict, List, Optional, Sequence

import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    ModelRequestConflict,
)
from roboflow_workflows.v2.blocks.models.detection import (
    ObjectDetectionBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    StepExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.pipelining import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.updates import compare_plans

from inference_models.models.base.object_detection import Detections
from tests.v2.support.compilation.fixture_blocks import Gate

MODES = ("run", "phases")
WAIT = 10.0


class RequestStampedModel:
    """Two boxes per image: a fixed one and one scored by image brightness."""

    def __init__(self, request: ModelRequest):
        self.request = request
        self.class_names = [f"{request}-class"]
        self.batch_sizes: List[int] = []
        self.closed = False

    def pre_process(self, images: List[torch.Tensor], input_color_format: str):
        assert input_color_format == "rgb"
        self.batch_sizes.append(len(images))
        batch = torch.stack([image.float().mean() / 255.0 for image in images])
        metadata = [tuple(image.shape[1:]) for image in images]

        return batch, metadata

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch * 1.0

    def post_process(self, raw: torch.Tensor, metadata: list, confidence: float):
        predictions = []
        for brightness, (height, width) in zip(raw.tolist(), metadata):
            scores = torch.tensor([0.9, round(brightness, 3)])
            keep = scores >= confidence
            predictions.append(
                Detections(
                    xyxy=torch.tensor([[0.0, 0.0, width, height]] * 2)[keep],
                    class_id=torch.tensor([0, 0])[keep],
                    confidence=scores[keep],
                    image_metadata={"served_by": str(self.request)},
                )
            )

        return predictions

    def close(self) -> None:
        self.closed = True


class FakeLoader:
    """Builds one ``RequestStampedModel`` per call; ``fail`` ids raise once."""

    def __init__(self, *, fail: Optional[List[str]] = None):
        self.fail = list(fail or [])
        self.loaded: List[ModelRequest] = []
        self.models: List[RequestStampedModel] = []

    def __call__(self, request: ModelRequest) -> RequestStampedModel:
        self.loaded.append(request)
        if request.model_id in self.fail:
            self.fail.remove(request.model_id)
            raise OSError(f"no weights for {request.model_id}")
        model = RequestStampedModel(request)
        self.models.append(model)

        return model


def _catalogue() -> Catalogue:
    catalogue = Catalogue(
        [ObjectDetectionBlock, Gate],
        kinds=[IMAGE_KIND, OBJECT_DETECTION_PREDICTION_KIND, FLOAT_KIND, STRING_KIND],
    )

    return catalogue


def _definition(
    *,
    per_image_model: bool = False,
    per_image_package: bool = False,
    gated: bool = False,
    model_package_id: Any = "$inputs.model_package_id",
) -> Dict[str, Any]:
    per_image = per_image_model or per_image_package or gated
    image_input = "WorkflowBatchInput" if per_image else "WorkflowParameter"
    model_input = "WorkflowBatchInput" if per_image_model else "WorkflowParameter"
    package_input = "WorkflowBatchInput" if per_image_package else "WorkflowParameter"
    step = {
        "type": "v2/object_detection@v1",
        "name": "detector",
        "image": "$inputs.image",
        "model_id": "$inputs.model_id",
        "confidence": 0.5,
    }
    if model_package_id is not None:
        step["model_package_id"] = model_package_id
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": image_input, "name": "image", "kind": ["image"]},
            {"type": model_input, "name": "model_id", "kind": ["string"]},
            {"type": package_input, "name": "model_package_id", "kind": ["string"]},
        ],
        "steps": [step],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.detector.predictions",
            }
        ],
    }
    if gated:
        definition["inputs"].append({"type": "WorkflowBatchInput", "name": "keep"})
        definition["steps"].insert(
            0,
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.detector"],
            },
        )

    return definition


def _plan(mode: str = "run", **definition_options: Any):
    plan = compile_workflow(
        _definition(**definition_options),
        catalogue=_catalogue(),
        options=CompileOptions(block_execution=mode),
    )

    return plan


def _image(
    *, value: int, image_id: str, height: int = 20, width: int = 30
) -> ImageData:
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _served(rows: List[Dict[str, Any]]) -> List[Optional[str]]:
    served = [
        (
            row["predictions"].image_metadata["served_by"]
            if row["predictions"] is not None
            else None
        )
        for row in rows
    ]

    return served


def _parents(rows: List[Dict[str, Any]]) -> List[Optional[str]]:
    parents = [
        (
            row["predictions"].image_metadata["parent_id"]
            if row["predictions"] is not None
            else None
        )
        for row in rows
    ]

    return parents


def _images(*image_ids: str) -> List[ImageData]:
    # Each image has its own height, so a row served another row's pixels
    # would show another box.
    images = [
        _image(value=200, image_id=image_id, height=10 + position)
        for position, image_id in enumerate(image_ids)
    ]

    return images


def _heights(rows: List[Dict[str, Any]]) -> List[Optional[float]]:
    heights = [
        (
            row["predictions"].xyxy.tolist()[0][3]
            if row["predictions"] is not None
            else None
        )
        for row in rows
    ]

    return heights


def _batch_sizes(loader: "FakeLoader") -> Dict[str, List[int]]:
    sizes = {str(model.request): model.batch_sizes for model in loader.models}

    return sizes


def _cause_of(error: BaseException, kind: type) -> BaseException:
    cause = error
    while cause is not None and not isinstance(cause, kind):
        cause = cause.__cause__
    assert cause is not None, f"no {kind.__name__} in the cause chain of {error!r}"

    return cause


@pytest.mark.parametrize("mode", MODES)
def test_selected_model_and_package_load_once_and_are_reused(mode: str) -> None:
    loader = FakeLoader()
    cache = ModelCache(loader)
    plan = _plan(mode)

    session = plan.create_session({"models": cache})
    assert loader.loaded == []  # compiling and constructing load nothing
    first = session.run(
        {
            "image": _image(value=200, image_id="a"),
            "model_id": "yolo",
            "model_package_id": "pkg-a",
        }
    ).rows()
    second = session.run(
        {
            "image": _image(value=200, image_id="b"),
            "model_id": "yolo",
            "model_package_id": "pkg-a",
        }
    ).rows()
    other = session.run(
        {
            "image": _image(value=200, image_id="c"),
            "model_id": "yolo",
            "model_package_id": "pkg-b",
        }
    ).rows()

    assert loader.loaded == [
        ModelRequest("yolo", "pkg-a"),
        ModelRequest("yolo", "pkg-b"),
    ]
    assert _served(first + second + other) == [
        "yolo (package 'pkg-a')",
        "yolo (package 'pkg-a')",
        "yolo (package 'pkg-b')",
    ]
    assert first[0]["predictions"].image_metadata["class_names"] == {
        0: "yolo (package 'pkg-a')-class"
    }


@pytest.mark.parametrize("mode", MODES)
def test_per_image_models_are_grouped_and_scattered_back_in_row_order(
    mode: str,
) -> None:
    loader = FakeLoader()
    session = _plan(mode, per_image_model=True).create_session(
        {"models": ModelCache(loader)}
    )
    images = [
        _image(value=200, image_id="a"),
        _image(value=20, image_id="b", height=10),
        _image(value=150, image_id="c", width=40),
        _image(value=255, image_id="d"),
    ]

    rows = session.run(
        {
            "image": images,
            "model_id": ["yolo", "rfdetr", "yolo", "rfdetr"],
            "model_package_id": "pkg-a",
        }
    ).rows()

    yolo, rfdetr = loader.models
    assert [yolo.request, rfdetr.request] == [
        ModelRequest("yolo", "pkg-a"),
        ModelRequest("rfdetr", "pkg-a"),
    ]
    assert yolo.batch_sizes == [2] and rfdetr.batch_sizes == [2]
    predictions = [row["predictions"] for row in rows]
    assert [p.image_metadata["parent_id"] for p in predictions] == ["a", "b", "c", "d"]
    assert _served(rows) == [
        "yolo (package 'pkg-a')",
        "rfdetr (package 'pkg-a')",
        "yolo (package 'pkg-a')",
        "rfdetr (package 'pkg-a')",
    ]
    # Pixels decide each row's second box: image "b" is dark.
    assert [len(p) for p in predictions] == [2, 1, 2, 2]
    assert predictions[1].xyxy.tolist() == [[0.0, 0.0, 30.0, 10.0]]
    assert predictions[2].xyxy.tolist()[0] == [0.0, 0.0, 40.0, 20.0]


def test_a_constant_request_over_a_batch_is_one_model_call() -> None:
    loader = FakeLoader()
    session = _plan(per_image_model=True).create_session({"models": ModelCache(loader)})

    session.run(
        {
            "image": [_image(value=v, image_id=str(v)) for v in (10, 20, 30)],
            "model_id": ["yolo", "yolo", "yolo"],
            "model_package_id": "pkg-a",
        }
    )

    (model,) = loader.models
    assert model.batch_sizes == [3]


def test_unset_package_requests_the_loader_default() -> None:
    loader = FakeLoader()
    session = _plan(model_package_id=None).create_session(
        {"models": ModelCache(loader)}
    )

    rows = session.run({"image": _image(value=200, image_id="a"), "model_id": "yolo"})

    assert loader.loaded == [ModelRequest("yolo")]
    assert _served(rows.rows()) == ["yolo"]


def test_sessions_sharing_a_cache_share_its_models() -> None:
    loader = FakeLoader()
    cache = ModelCache(loader)
    plan = _plan()
    inputs = {"model_id": "yolo", "model_package_id": "pkg-a"}

    for image_id in ("a", "b"):
        session = plan.create_session({"models": cache})
        session.run({"image": _image(value=200, image_id=image_id), **inputs})
        session.close()

    assert loader.loaded == [ModelRequest("yolo", "pkg-a")]


def test_close_and_reset_keep_cached_and_supplied_models() -> None:
    supplied = RequestStampedModel(ModelRequest("supplied", "pkg-s"))
    loader = FakeLoader()
    cache = ModelCache(loader, supplied={ModelRequest("supplied", "pkg-s"): supplied})
    session = _plan().create_session({"models": cache})
    session.run(
        {
            "image": _image(value=200, image_id="a"),
            "model_id": "yolo",
            "model_package_id": "pkg-a",
        }
    )
    session.run(
        {
            "image": _image(value=200, image_id="b"),
            "model_id": "supplied",
            "model_package_id": "pkg-s",
        }
    )

    session.update(_plan(), reset=True)
    session.run(
        {
            "image": _image(value=200, image_id="c"),
            "model_id": "yolo",
            "model_package_id": "pkg-a",
        }
    )
    session.close()

    (loaded,) = loader.models
    assert not loaded.closed and not supplied.closed
    assert loaded.batch_sizes == [1, 1]
    assert cache.get(ModelRequest("yolo", "pkg-a")) is loaded
    assert cache.get(ModelRequest("supplied", "pkg-s")) is supplied


def test_a_package_request_against_a_bare_mapping_fails_without_a_model_call() -> None:
    model = RequestStampedModel(ModelRequest("yolo"))
    session = _plan().create_session({"models": {"yolo": model}})

    with pytest.raises(StepExecutionError) as caught:
        session.run(
            {
                "image": _image(value=200, image_id="a"),
                "model_id": "yolo",
                "model_package_id": "pkg-a",
            }
        )

    conflict = _cause_of(caught.value, ModelRequestConflict)
    assert "yolo (package 'pkg-a')" in str(conflict)
    assert "unverified" in str(conflict)
    assert model.batch_sizes == []


def test_a_mapping_serves_its_request_bound_models() -> None:
    model = RequestStampedModel(ModelRequest("yolo", "pkg-a"))
    session = _plan().create_session({"models": {ModelRequest("yolo", "pkg-a"): model}})

    rows = session.run(
        {
            "image": _image(value=200, image_id="a"),
            "model_id": "yolo",
            "model_package_id": "pkg-a",
        }
    ).rows()

    assert _served(rows) == ["yolo (package 'pkg-a')"]
    assert model.batch_sizes == [1]


def test_one_bad_row_request_fails_the_batch_before_any_model_call() -> None:
    loader = FakeLoader()
    cache = ModelCache(
        loader, supplied={"yolo": RequestStampedModel(ModelRequest("yolo"))}
    )
    session = _plan(per_image_model=True).create_session({"models": cache})

    with pytest.raises(StepExecutionError) as caught:
        session.run(
            {
                "image": [
                    _image(value=200, image_id="a"),
                    _image(value=9, image_id="b"),
                ],
                "model_id": ["rfdetr", "yolo"],
                "model_package_id": "pkg-a",
            }
        )

    _cause_of(caught.value, ModelRequestConflict)
    (loaded,) = loader.models
    assert loaded.request == ModelRequest("rfdetr", "pkg-a")
    assert loaded.batch_sizes == []


def test_a_failed_load_names_the_request_and_the_next_run_retries() -> None:
    loader = FakeLoader(fail=["yolo"])
    session = _plan().create_session({"models": ModelCache(loader)})
    inputs = {
        "image": _image(value=200, image_id="a"),
        "model_id": "yolo",
        "model_package_id": "pkg-a",
    }

    with pytest.raises(StepExecutionError) as caught:
        session.run(inputs)
    rows = session.run(inputs).rows()

    assert "no weights for yolo" in str(_cause_of(caught.value, OSError))
    assert _served(rows) == ["yolo (package 'pkg-a')"]
    assert loader.loaded == [ModelRequest("yolo", "pkg-a")] * 2


def test_discovery_reports_a_selected_model_as_known_only_at_run_time() -> None:
    resources = discover_workload(_plan()).resources

    assert resources.items == [] and not resources.complete
    (problem,) = resources.unknown_reasons
    assert problem.details["selector"] == "$inputs.model_id"
    (declared,) = ObjectDetectionBlock.discover_dependent_resources(
        ObjectDetectionBlock.Params(
            image="$inputs.image", model_id="yolo", model_package_id="pkg-a"
        )
    )
    assert declared.identifier == "yolo"
    assert declared.details["model_package_id"] == "pkg-a"


def test_changing_a_literal_package_requires_a_reset() -> None:
    old = _plan(model_package_id="pkg-a")
    new = _plan(model_package_id="pkg-b")

    diff = compare_plans(old, new)

    assert diff.kind == "reset"
    assert [(change.name, change.reason) for change in diff.changes] == [
        ("$steps.detector", "params_changed")
    ]


# --- per-image packages (Client C-1, Tester T1) ----------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "packages, loaded, served",
    [
        (
            [None, "trt"],
            [ModelRequest("yolo"), ModelRequest("yolo", "trt")],
            ["yolo", "yolo (package 'trt')"],
        ),
        ([None, None], [ModelRequest("yolo")], ["yolo", "yolo"]),
        (
            ["onnx", "trt"],
            [ModelRequest("yolo", "onnx"), ModelRequest("yolo", "trt")],
            ["yolo (package 'onnx')", "yolo (package 'trt')"],
        ),
    ],
    ids=["none-and-pinned", "all-none", "all-pinned"],
)
def test_a_per_image_package_of_none_lets_the_loader_choose_for_that_image(
    mode: str,
    packages: List[Optional[str]],
    loaded: List[ModelRequest],
    served: List[str],
) -> None:
    loader = FakeLoader()
    session = _plan(mode, per_image_package=True).create_session(
        {"models": ModelCache(loader)}
    )

    rows = session.run(
        {"image": _images("a", "b"), "model_id": "yolo", "model_package_id": packages}
    ).rows()

    assert loader.loaded == loaded
    assert _served(rows) == served
    assert _parents(rows) == ["a", "b"]
    assert _heights(rows) == [10.0, 11.0]
    assert sum(len(model.batch_sizes) for model in loader.models) == len(loaded)


@pytest.mark.parametrize("mode", MODES)
def test_per_image_packages_group_rows_by_request_in_row_order(mode: str) -> None:
    # Tester T1: the per-row package path, with no None involved.
    loader = FakeLoader()
    session = _plan(mode, per_image_package=True).create_session(
        {"models": ModelCache(loader)}
    )

    rows = session.run(
        {
            "image": _images("a", "b", "c"),
            "model_id": "yolo",
            "model_package_id": ["pkg-a", "pkg-b", "pkg-a"],
        }
    ).rows()

    assert loader.loaded == [
        ModelRequest("yolo", "pkg-a"),
        ModelRequest("yolo", "pkg-b"),
    ]
    assert _batch_sizes(loader) == {
        "yolo (package 'pkg-a')": [2],
        "yolo (package 'pkg-b')": [1],
    }
    assert _served(rows) == [
        "yolo (package 'pkg-a')",
        "yolo (package 'pkg-b')",
        "yolo (package 'pkg-a')",
    ]
    assert _parents(rows) == ["a", "b", "c"]
    assert _heights(rows) == [10.0, 11.0, 12.0]


def test_too_few_per_image_packages_fail_before_any_model_call() -> None:
    # Input validation rejects the lengths before the step runs.
    loader = FakeLoader()
    session = _plan(per_image_package=True).create_session(
        {"models": ModelCache(loader)}
    )

    with pytest.raises(WorkflowInputError) as caught:
        session.run(
            {
                "image": _images("a", "b", "c"),
                "model_id": "yolo",
                "model_package_id": ["pkg-a", "pkg-b"],
            }
        )

    assert "model_package_id" in str(caught.value)
    assert loader.loaded == []


def test_mixed_model_ids_and_packages_each_get_their_own_request() -> None:
    loader = FakeLoader()
    session = _plan(
        "phases", per_image_model=True, per_image_package=True
    ).create_session({"models": ModelCache(loader)})

    rows = session.run(
        {
            "image": _images("a", "b", "c", "d"),
            "model_id": ["yolo", "rfdetr", "yolo", "rfdetr"],
            "model_package_id": [None, None, "trt", None],
        }
    ).rows()

    assert loader.loaded == [
        ModelRequest("yolo"),
        ModelRequest("rfdetr"),
        ModelRequest("yolo", "trt"),
    ]
    assert _batch_sizes(loader) == {
        "yolo": [1],
        "rfdetr": [2],
        "yolo (package 'trt')": [1],
    }
    assert _served(rows) == ["yolo", "rfdetr", "yolo (package 'trt')", "rfdetr"]
    assert _parents(rows) == ["a", "b", "c", "d"]
    assert _heights(rows) == [10.0, 11.0, 12.0, 13.0]


@pytest.mark.parametrize("mode", MODES)
def test_a_gated_image_is_skipped_while_none_packages_still_run(mode: str) -> None:
    loader = FakeLoader()
    session = _plan(mode, per_image_package=True, gated=True).create_session(
        {"models": ModelCache(loader)}
    )

    rows = session.run(
        {
            "image": _images("a", "b", "c", "d"),
            "model_id": "yolo",
            "model_package_id": [None, "trt", None, "trt"],
            "keep": [True, False, True, True],
        }
    ).rows()

    assert loader.loaded == [ModelRequest("yolo"), ModelRequest("yolo", "trt")]
    assert _batch_sizes(loader) == {"yolo": [2], "yolo (package 'trt')": [1]}
    assert _served(rows) == ["yolo", None, "yolo", "yolo (package 'trt')"]
    assert _parents(rows) == ["a", None, "c", "d"]
    assert _heights(rows) == [10.0, None, 12.0, 13.0]


def test_a_per_image_model_id_of_none_skips_that_image() -> None:
    loader = FakeLoader()
    session = _plan(per_image_model=True).create_session({"models": ModelCache(loader)})

    rows = session.run(
        {
            "image": _images("a", "b"),
            "model_id": [None, "yolo"],
            "model_package_id": "pkg-a",
        }
    ).rows()

    assert loader.loaded == [ModelRequest("yolo", "pkg-a")]
    assert _served(rows) == [None, "yolo (package 'pkg-a')"]
    assert _parents(rows) == [None, "b"]


def test_a_scalar_none_package_parameter_still_lets_the_loader_choose() -> None:
    loader = FakeLoader()
    session = _plan(per_image_model=True).create_session({"models": ModelCache(loader)})

    rows = session.run(
        {"image": _images("a", "b"), "model_id": ["yolo"] * 2, "model_package_id": None}
    ).rows()

    assert loader.loaded == [ModelRequest("yolo")]
    assert _served(rows) == ["yolo", "yolo"]


@pytest.mark.parametrize(
    "bad, error",
    # The block's own validator rejects ""; the input's kind rejects a number.
    [("", StepExecutionError), (7, WorkflowInputError)],
    ids=["empty", "number"],
)
def test_an_invalid_per_image_package_names_the_parameter_and_row(
    bad: Any, error: type
) -> None:
    loader = FakeLoader()
    session = _plan(per_image_package=True).create_session(
        {"models": ModelCache(loader)}
    )

    with pytest.raises(error) as caught:
        session.run(
            {
                "image": _images("a", "b"),
                "model_id": "yolo",
                "model_package_id": [None, bad],
            }
        )

    message = str(caught.value)
    assert "model_package_id" in message and "[1]" in message
    assert loader.loaded == []


def test_a_bare_supplied_model_serves_none_and_rejects_a_pinned_package() -> None:
    supplied = RequestStampedModel(ModelRequest("yolo"))
    loader = FakeLoader()
    cache = ModelCache(loader, supplied={"yolo": supplied})
    session = _plan(per_image_package=True).create_session({"models": cache})
    inputs = {"image": _images("a", "b"), "model_id": "yolo"}

    rows = session.run({**inputs, "model_package_id": [None, None]}).rows()
    with pytest.raises(StepExecutionError) as caught:
        session.run({**inputs, "model_package_id": [None, "pkg-a"]})

    assert _served(rows) == ["yolo", "yolo"]
    assert supplied.batch_sizes == [2]  # the conflicting run called no model
    _cause_of(caught.value, ModelRequestConflict)
    assert loader.loaded == []


def test_pipelined_runs_keep_each_runs_packages_and_load_each_request_once() -> None:
    loader = FakeLoader()
    session = _plan("phases", per_image_package=True).create_session(
        {"models": ModelCache(loader)}
    )
    submissions: Sequence[List[Optional[str]]] = (
        [None, "trt"],
        ["trt", None],
        [None, None],
    )

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        futures = [
            pipeline.submit(
                {
                    "image": _images(f"{run}a", f"{run}b"),
                    "model_id": "yolo",
                    "model_package_id": list(packages),
                }
            )
            for run, packages in enumerate(submissions)
        ]
        results = [future.result(timeout=WAIT).rows() for future in futures]

    assert loader.loaded == [ModelRequest("yolo"), ModelRequest("yolo", "trt")]
    assert [_served(rows) for rows in results] == [
        ["yolo", "yolo (package 'trt')"],
        ["yolo (package 'trt')", "yolo"],
        ["yolo", "yolo"],
    ]
    assert [_parents(rows) for rows in results] == [
        ["0a", "0b"],
        ["1a", "1b"],
        ["2a", "2b"],
    ]
