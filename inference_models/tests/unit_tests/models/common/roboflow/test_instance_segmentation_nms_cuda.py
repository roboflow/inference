"""CUDA tests for ``run_nms_for_instance_segmentation`` ordered-index selection.

Same finite case table as the CPU tests, run on CUDA against the frozen
reference on the same backend. Stream tests use explicit input-ready joins and
``record_stream`` to protect storage used on non-allocating streams.
"""

import threading

import pytest
import torch

from inference_models.models.common.roboflow.post_processing import (
    run_nms_for_instance_segmentation,
)
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

pytestmark = [
    pytest.mark.gpu_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

PARITY_NAMES = [case.name for case in parity_cases()]
ERROR_NAMES = [case.name for case in error_cases()]
STREAM_CASES = [
    "B1_C3_defaults",
    "B8_C3_transposed_block_options",
    "B2_C3_strided_per_class_agnostic_xyxy",
    "B8_C80_per_class",
]


def _case(table, name):
    case = next(case for case in table("cuda") if case.name == name)
    return case


@pytest.mark.parametrize("name", PARITY_NAMES)
def test_matches_reference_bit_for_bit_on_cuda(name: str) -> None:
    case = _case(parity_cases, name)
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
        assert all(out.device.type == "cuda" for out in actual.value)
        assert not any(shares_storage(out, case.raw) for out in actual.value)


@pytest.mark.parametrize("name", ERROR_NAMES)
def test_preserves_preexisting_errors_on_cuda(name: str) -> None:
    case = _case(error_cases, name)

    expected = call_outcome(
        reference_run_nms_for_instance_segmentation, case.raw, case.options
    )
    actual = call_outcome(run_nms_for_instance_segmentation, case.raw, case.options)

    assert expected.kind == "error"
    assert_same_outcome(actual, expected)


def test_cuda_layouts_match_cpu_construction() -> None:
    for cpu_case, cuda_case in zip(parity_cases("cpu"), parity_cases("cuda")):
        cpu, cuda = cpu_case.raw, cuda_case.raw
        assert (
            cuda.dtype,
            tuple(cuda.shape),
            cuda.stride(),
            cuda.storage_offset(),
        ) == (
            cpu.dtype,
            tuple(cpu.shape),
            cpu.stride(),
            cpu.storage_offset(),
        ), cpu_case.name
        assert tensor_bytes(cuda) == tensor_bytes(cpu), cpu_case.name


def test_per_class_cpu_threshold_is_transferred_and_kept() -> None:
    case = _case(parity_cases, "B8_C80_per_class")
    threshold = case.options["conf_thresh"]
    before = (threshold.device.type, tensor_bytes(threshold))

    result = run_nms_for_instance_segmentation(case.raw, **case.options)

    assert (
        before[0] == "cpu"
        and (threshold.device.type, tensor_bytes(threshold)) == before
    )
    assert all(out.device.type == "cuda" for out in result)


def test_empty_results_use_default_dtype_on_cuda() -> None:
    for name in (
        "B1_C3_all_below_threshold",
        "B1_C3_rank2_threshold_all_false",
        "B1_C3_float64_all_below_threshold",
        "B1_C3_float16_all_below_threshold",
    ):
        case = _case(parity_cases, name)
        result = run_nms_for_instance_segmentation(case.raw, **case.options)
        assert (
            result[0].shape == (0, 38) and result[0].dtype == torch.get_default_dtype()
        )
        assert result[0].device == case.raw.device


@pytest.mark.parametrize("name", ["B1_C3_float16", "B1_C3_bfloat16"])
def test_half_precision_backend_domain_is_recorded(name: str, record_property) -> None:
    # Backend support differs by device; the parity test holds either way.
    case = _case(parity_cases, name)

    outcome = call_outcome(run_nms_for_instance_segmentation, case.raw, case.options)

    if outcome.kind == "ok":
        record_property(
            name, f"ok dtype={outcome.value[0].dtype} rows={outcome.value[0].shape[0]}"
        )
    else:
        record_property(name, f"error {outcome.value[0]}: {outcome.value[1][:200]}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_of_retained_values_match_on_cuda(dtype) -> None:
    base = make_raw(2, 40, 3, seed=6, dtype=dtype, device="cuda")
    base[:, 4 + 3, 0] = 0.0
    gradients = []
    for function in (
        reference_run_nms_for_instance_segmentation,
        run_nms_for_instance_segmentation,
    ):
        raw = base.clone().requires_grad_(True)
        result = function(raw, iou_thresh=0.5)
        weights = [
            torch.linspace(
                -1, 1, out.numel(), dtype=out.dtype, device=out.device
            ).reshape(out.shape)
            for out in result
        ]
        sum((out * weight).sum() for out, weight in zip(result, weights)).backward()
        gradients.append(tensor_bytes(raw.grad))

    assert gradients[0] == gradients[1]


def _run_on_stream(stream, case):
    with torch.cuda.stream(stream):
        case.raw.record_stream(stream)
        result = run_nms_for_instance_segmentation(case.raw, **case.options)
    return result


def test_two_streams_in_two_threads_match_serial() -> None:
    cases = [_case(parity_cases, name) for name in STREAM_CASES]
    serial = {
        case.name: [
            tensor_bytes(out)
            for out in run_nms_for_instance_segmentation(case.raw, **case.options)
        ]
        for case in cases
    }
    producer = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    barrier = threading.Barrier(2)
    results, failures = [], []

    def work(stream):
        try:
            stream.wait_stream(producer)  # inputs were written on the producer stream
            barrier.wait()
            for _ in range(3):
                for case in cases:
                    results.append((case.name, _run_on_stream(stream, case)))
            stream.synchronize()
        except Exception as error:  # surfaced below
            failures.append(error)

    threads = [threading.Thread(target=work, args=(stream,)) for stream in streams]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(results) == 2 * 3 * len(cases)
    for name, outputs in results:
        assert [tensor_bytes(out) for out in outputs] == serial[name]


def test_third_stream_reads_outputs_after_raw_owner_release() -> None:
    expected_case = _case(parity_cases, "B8_C3_transposed_block_options")
    expected = [
        tensor_bytes(out)
        for out in run_nms_for_instance_segmentation(
            expected_case.raw, **expected_case.options
        )
    ]
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(producer):
        raw = make_raw(8, 48, 3, seed=2, layout="transposed", device="cuda")
        outputs = run_nms_for_instance_segmentation(raw, **expected_case.options)
    del raw  # release the raw owner; its block returns to the producer pool
    with torch.cuda.stream(producer):
        junk = torch.full((8 * 39 * 48 * 4,), float("nan"), device="cuda")
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        for out in outputs:
            out.record_stream(consumer)
        copies = [out.clone() for out in outputs]
    consumer.synchronize()

    assert [tensor_bytes(copy) for copy in copies] == expected
    del junk
