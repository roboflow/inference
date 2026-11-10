"""CPU tests for ``run_nms_for_instance_segmentation`` ordered-index selection.

The frozen pre-change reference lives in ``instance_segmentation_nms_cases``.
Parity is complemented by hand-computed rows, NMS input order, ownership,
gradients, threads and the real dense/RLE/grid producers.
"""

import ast
import dataclasses
import hashlib
import inspect
import threading
import warnings

import numpy as np
import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow import post_processing
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    run_nms_for_instance_segmentation,
)
from inference_models.models.yolov8 import common as yolov8_common
from tests.unit_tests.models.common.roboflow.instance_segmentation_nms_cases import (
    assert_same_outcome,
    call_outcome,
    error_cases,
    make_raw,
    parity_cases,
    record_nms_inputs,
    reference_run_nms_for_instance_segmentation,
    shares_storage,
    tensor_bytes,
)

PARITY = {case.name: case for case in parity_cases()}
ERRORS = {case.name: case for case in error_cases()}


@pytest.mark.parametrize("name", list(PARITY))
def test_matches_reference_bit_for_bit(name: str) -> None:
    case = PARITY[name]
    raw_before = tensor_bytes(case.raw)

    with record_nms_inputs() as reference_calls:
        expected = call_outcome(
            reference_run_nms_for_instance_segmentation, case.raw, case.options
        )
    with record_nms_inputs() as calls:
        actual = call_outcome(run_nms_for_instance_segmentation, case.raw, case.options)

    assert_same_outcome(actual, expected)
    assert calls == reference_calls
    assert tensor_bytes(case.raw) == raw_before
    if actual.kind == "ok":
        assert not any(shares_storage(out, case.raw) for out in actual.value)


@pytest.mark.parametrize("name", list(ERRORS))
def test_preserves_preexisting_errors(name: str) -> None:
    case = ERRORS[name]

    expected = call_outcome(
        reference_run_nms_for_instance_segmentation, case.raw, case.options
    )
    actual = call_outcome(run_nms_for_instance_segmentation, case.raw, case.options)

    assert expected.kind == "error"
    assert_same_outcome(actual, expected)


def test_known_backend_outcomes_on_cpu() -> None:
    # CPU NMS rejects half precision; both versions must keep failing there.
    # The exception type is torchvision's: RuntimeError in 0.29,
    # NotImplementedError in 0.26.
    for name in ("B1_C3_float16", "B1_C3_bfloat16"):
        outcome = call_outcome(
            run_nms_for_instance_segmentation, PARITY[name].raw, PARITY[name].options
        )
        assert outcome.kind == "error", name
        assert outcome.value[0] in ("RuntimeError", "NotImplementedError"), name
    double = call_outcome(
        run_nms_for_instance_segmentation, PARITY["B1_C3_float64"].raw, {}
    )
    assert double.kind == "ok" and double.value[0].dtype == torch.float64


def test_rank2_threshold_reproduces_original_index_error() -> None:
    # The design's counterexample: the compacted path alone would accept this.
    outcome = call_outcome(
        run_nms_for_instance_segmentation,
        ERRORS["rank2_threshold_mixed"].raw,
        ERRORS["rank2_threshold_mixed"].options,
    )

    assert outcome.kind == "error" and outcome.value[0] == "IndexError"


def test_empty_results_use_default_dtype_on_output_device() -> None:
    for name in (
        "B1_C3_all_below_threshold",
        "B1_C3_rank2_threshold_all_false",
        "B1_C3_float64_all_below_threshold",
        "B1_C3_float16_all_below_threshold",
    ):
        result = run_nms_for_instance_segmentation(
            PARITY[name].raw, **PARITY[name].options
        )
        assert len(result) == 1 and result[0].shape == (0, 38)
        assert (
            result[0].dtype == torch.get_default_dtype()
            and result[0].device.type == "cpu"
        )
    assert run_nms_for_instance_segmentation(PARITY["B0_C3"].raw) == []


def _hand_built_raw() -> torch.Tensor:
    raw = torch.zeros((1, 4 + 2 + 32, 6))
    boxes = [
        (10, 10, 4, 4),
        (10, 10, 4, 4),
        (10, 10, 4, 4),
        (50, 50, 4, 4),
        (50, 50, 4, 4),
        (90, 90, 4, 4),
    ]
    scores = [(0.9, 0.0), (0.8, 0.0), (0.0, 0.7), (0.0, 0.25), (0.6, 0.0), (0.1, 0.1)]
    for anchor, (box, score) in enumerate(zip(boxes, scores)):
        raw[0, :4, anchor] = torch.tensor(box, dtype=torch.float32)
        raw[0, 4:6, anchor] = torch.tensor(score)
        raw[0, 6:, anchor] = torch.arange(32) * 0.5 + anchor
    return raw


def _expected_row(raw: torch.Tensor, anchor: int, class_id: int) -> torch.Tensor:
    x, y, w, h = raw[0, :4, anchor]
    head = torch.stack(
        [x - w / 2, y - h / 2, x + w / 2, y + h / 2, raw[0, 4 + class_id, anchor]]
    )
    row = torch.cat([head, torch.tensor([float(class_id)]), raw[0, 6:, anchor]])
    return row


@pytest.mark.parametrize(
    "class_agnostic,kept",
    [(False, [(0, 0), (2, 1), (4, 0)]), (True, [(0, 0), (4, 0)])],
)
def test_hand_computed_rows_and_nms_input_order(class_agnostic, kept) -> None:
    raw = _hand_built_raw()

    with record_nms_inputs() as calls:
        result = run_nms_for_instance_segmentation(
            raw, conf_thresh=0.25, iou_thresh=0.5, class_agnostic=class_agnostic
        )

    expected = torch.stack(
        [_expected_row(raw, anchor, class_id) for anchor, class_id in kept]
    )
    assert torch.equal(result[0], expected)
    # Anchor 3 sits exactly at the threshold and anchor 5 below it; the rest
    # reach NMS in ascending anchor order.
    passing_scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    assert len(calls) == 1
    assert calls[0]["scores"][2] == _digest(passing_scores)


def _digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256(tensor_bytes(tensor)).hexdigest()

    return digest


def test_outputs_survive_raw_mutation_and_repeat_calls() -> None:
    raw = make_raw(2, 40, 3, seed=6, layout="strided")
    first = run_nms_for_instance_segmentation(raw)
    snapshot = [tensor_bytes(out) for out in first]

    second = run_nms_for_instance_segmentation(raw)
    raw.fill_(float("nan"))

    assert [tensor_bytes(out) for out in second] == snapshot
    assert [tensor_bytes(out) for out in first] == snapshot


def test_per_class_threshold_tensor_is_not_modified() -> None:
    case = PARITY["B8_C80_per_class"]
    threshold = case.options["conf_thresh"]
    before = (threshold.device, tensor_bytes(threshold))

    run_nms_for_instance_segmentation(case.raw, **case.options)

    assert (threshold.device, tensor_bytes(threshold)) == before


def test_read_only_numpy_backed_input() -> None:
    array = make_raw(1, 64, 3, seed=1).numpy().copy()
    array.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = torch.from_numpy(array)

    expected = call_outcome(reference_run_nms_for_instance_segmentation, raw, {})
    actual = call_outcome(run_nms_for_instance_segmentation, raw, {})

    assert_same_outcome(actual, expected)
    assert not array.flags.writeable


def test_out_of_range_per_class_threshold_error_on_cpu() -> None:
    # CPU only: on CUDA this indexes out of bounds inside a kernel.
    raw = make_raw(1, 64, 3, seed=1)
    options = {"conf_thresh": torch.tensor([0.25, 0.25])}

    expected = call_outcome(reference_run_nms_for_instance_segmentation, raw, options)
    actual = call_outcome(run_nms_for_instance_segmentation, raw, options)

    assert expected.kind == "error"
    assert_same_outcome(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_of_retained_values_match(dtype) -> None:
    base = make_raw(2, 40, 3, seed=6, dtype=dtype)
    base[:, 4 + 3, 0] = 0.0  # keep the gradient finite
    gradients = []
    for function in (
        reference_run_nms_for_instance_segmentation,
        run_nms_for_instance_segmentation,
    ):
        raw = base.clone().requires_grad_(True)
        result = function(raw, iou_thresh=0.5)
        weights = [
            torch.linspace(-1, 1, out.numel(), dtype=out.dtype).reshape(out.shape)
            for out in result
        ]
        sum((out * weight).sum() for out, weight in zip(result, weights)).backward()
        gradients.append(tensor_bytes(raw.grad))

    assert gradients[0] == gradients[1]
    assert any(byte != 0 for byte in gradients[0])


def test_threads_with_distinct_and_shared_inputs_match_serial() -> None:
    names = [
        "B1_C3_defaults",
        "B8_C3_transposed_block_options",
        "B2_C3_strided_per_class_agnostic_xyxy",
        "B4_C3_expanded",
    ]
    cases = [PARITY[name] for name in names]
    serial = {
        case.name: call_outcome(
            run_nms_for_instance_segmentation, case.raw, case.options
        )
        for case in cases
    }
    function_state = (
        post_processing.run_nms_for_instance_segmentation,
        run_nms_for_instance_segmentation.__code__,
        run_nms_for_instance_segmentation.__defaults__,
        set(vars(post_processing)),
    )
    barrier = threading.Barrier(len(cases) * 2)
    results, failures = [], []

    def work(case):
        try:
            barrier.wait()
            for _ in range(5):
                results.append(
                    (
                        case.name,
                        call_outcome(
                            run_nms_for_instance_segmentation, case.raw, case.options
                        ),
                    )
                )
        except Exception as error:  # surfaced below
            failures.append(error)

    # Two threads per case: distinct inputs across cases, shared read-only input within one.
    threads = [
        threading.Thread(target=work, args=(case,)) for case in cases for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(results) == len(cases) * 2 * 5
    for name, outcome in results:
        assert_same_outcome(outcome, serial[name])
    assert function_state == (
        post_processing.run_nms_for_instance_segmentation,
        run_nms_for_instance_segmentation.__code__,
        run_nms_for_instance_segmentation.__defaults__,
        set(vars(post_processing)),
    )


def _function_node(source: str, name: str) -> ast.FunctionDef:
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return node


def test_only_selection_and_docstring_changed() -> None:
    from tests.unit_tests.models.common.roboflow import (
        instance_segmentation_nms_cases as cases_module,
    )

    new = _function_node(
        inspect.getsource(post_processing), "run_nms_for_instance_segmentation"
    )
    old = _function_node(
        inspect.getsource(cases_module), "reference_run_nms_for_instance_segmentation"
    )
    assert ast.dump(new.args) == ast.dump(old.args) and ast.dump(
        new.returns
    ) == ast.dump(old.returns)

    new_loop, old_loop = new.body[-2], old.body[-2]
    assert [ast.dump(s) for s in new.body[1:-2]] == [
        ast.dump(s) for s in old.body[1:-2]
    ]
    assert ast.dump(new.body[-1]) == ast.dump(old.body[-1])
    # Loop body: identical before the selection and after the four gathers.
    assert [ast.dump(s) for s in new_loop.body[:5]] == [
        ast.dump(s) for s in old_loop.body[:5]
    ]
    assert [ast.dump(s) for s in new_loop.body[10:]] == [
        ast.dump(s) for s in old_loop.body[10:]
    ]
    assert len(new_loop.body) == len(old_loop.body)
    rename = {"selection": "mask"}

    def normalized(statement):
        tree = ast.parse(ast.unparse(statement))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in rename:
                node.id = rename[node.id]
        return ast.dump(tree)

    assert [normalized(s) for s in new_loop.body[6:10]] == [
        normalized(s) for s in old_loop.body[6:10]
    ]
    fallback = new_loop.body[5].orelse
    assert ast.dump(fallback[1]) == ast.dump(old_loop.body[5])


def _producer_inputs():
    generator = torch.Generator().manual_seed(11)
    raw = torch.zeros((2, 4 + 3 + 32, 24))
    raw[:, 0] = torch.rand((2, 24), generator=generator) * 48 + 8
    raw[:, 1] = torch.rand((2, 24), generator=generator) * 16 + 8
    raw[:, 2:4] = torch.rand((2, 2, 24), generator=generator) * 20 + 4
    raw[:, 4:7] = torch.rand((2, 3, 24), generator=generator)
    raw[:, 7:] = torch.randn((2, 32, 24), generator=generator)
    raw[1, 4:7] = 0.0  # second image selects nothing
    protos = torch.randn((2, 32, 8, 16), generator=generator)
    metadata = PreProcessingMetadata(
        original_size=ImageDimensions(height=60, width=80),
        size_after_pre_processing=ImageDimensions(height=60, width=80),
        inference_size=ImageDimensions(height=32, width=64),
        nonsquare_intermediate_size=None,
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        scale_width=64 / 80,
        scale_height=32 / 60,
        static_crop_offset=StaticCropOffset(
            offset_x=0, offset_y=0, crop_width=80, crop_height=60
        ),
    )
    return raw, protos, [metadata, metadata]


def _assert_same_structure(left, right, path="result") -> None:
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        assert (left.dtype, left.device, left.shape) == (
            right.dtype,
            right.device,
            right.shape,
        ), path
        assert tensor_bytes(left) == tensor_bytes(right), path
    elif dataclasses.is_dataclass(left):
        for field in dataclasses.fields(left):
            _assert_same_structure(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{path}.{field.name}",
            )
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_same_structure(a, b, f"{path}[{index}]")
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_same_structure(left[key], right[key], f"{path}[{key!r}]")
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes(), path
    else:
        assert left == right, path


@pytest.mark.parametrize(
    "producer", ["prepare_dense_masks", "prepare_rle_masks", "prepare_grid_masks"]
)
@pytest.mark.parametrize("smoothing", [False, True])
def test_real_nms_feeds_producers_identically(producer: str, smoothing: bool) -> None:
    raw, protos, metadata = _producer_inputs()
    outputs = []
    for function in (
        reference_run_nms_for_instance_segmentation,
        run_nms_for_instance_segmentation,
    ):
        selected = function(raw, conf_thresh=0.25, iou_thresh=0.5, max_detections=300)
        outputs.append(
            getattr(yolov8_common, producer)(
                nms_results=selected,
                protos=protos,
                pre_processing_meta=metadata,
                masks_smoothing_enabled=smoothing,
                masks_binarization_threshold=0.5,
            )
        )

    assert len(outputs[0][0].confidence) > 1 and len(outputs[0][1].confidence) == 0
    _assert_same_structure(outputs[1], outputs[0])
