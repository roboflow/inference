"""Native producer and CPU dispatch coverage; no weights or GPU execution."""

import ast
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import inference_models
from inference_models import InstanceDetections, InstanceSegmentationGrid
from inference_models.entities import ImageDimensions
from inference_models.errors import ModelInputError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.rfdetr.class_remapping import ClassesReMapping

FAMILIES = ("yolov5", "yolov7", "yolov8", "yolo26", "yolact", "rfdetr")
MODEL_ROOT = Path(inference_models.__file__).parent / "models"
BACKENDS = tuple(
    path.relative_to(MODEL_ROOT).as_posix()
    for family in FAMILIES
    for path in sorted((MODEL_ROOT / family).glob("*instance_segmentation*.py"))
)


def _meta(*, padding=(0, 0, 0, 0), crop=(0, 0, 80, 60), nonsquare=False):
    left, top, right, bottom = padding
    x, y, width, height = crop
    effective = ImageDimensions(height=48 if nonsquare else 32, width=64)
    return PreProcessingMetadata(
        original_size=ImageDimensions(height=60, width=80),
        size_after_pre_processing=ImageDimensions(height=height, width=width),
        inference_size=ImageDimensions(height=32, width=64),
        nonsquare_intermediate_size=effective if nonsquare else None,
        pad_left=left,
        pad_top=top,
        pad_right=right,
        pad_bottom=bottom,
        scale_width=(effective.width - left - right) / width,
        scale_height=(effective.height - top - bottom) / height,
        static_crop_offset=StaticCropOffset(
            offset_x=x, offset_y=y, crop_width=width, crop_height=height
        ),
    )


def _refuse(*args, **kwargs):
    raise AssertionError("grid producer called dense alignment or RLE encoding")


def _rows():
    return torch.tensor(
        [[4.5, 4, 52, 28, 0.8, 2, 1.0, -0.2], [0, 0, 64, 32, 0.9, 1, -0.5, 1.0]]
    )


def _prototypes():
    return torch.linspace(-2, 2, 2 * 8 * 16).reshape(2, 8, 16)


def _prepare(family, module, rows, protos, metadata, *, smoothing=False, threshold=0):
    kwargs = {"pre_processing_meta": metadata}
    kwargs["filtered_results" if family == "yolo26" else "nms_results"] = rows
    kwargs["all_proto_data" if family == "yolact" else "protos"] = (
        protos.permute(0, 2, 3, 1) if family == "yolact" else protos
    )
    if family == "yolov8":
        kwargs.update(
            masks_smoothing_enabled=smoothing,
            masks_binarization_threshold=threshold,
        )
    return module.prepare_grid_masks(**kwargs)


@pytest.mark.parametrize("family", FAMILIES[:-1])
@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_prototype_grids_preserve_native_scores_and_rows(
    family, smoothing, dtype, monkeypatch
):
    module = import_module(f"inference_models.models.{family}.common")
    monkeypatch.setattr(module, "align_instance_segmentation_results", _refuse)
    monkeypatch.setattr(
        module, "align_instance_segmentation_results_to_rle_masks", _refuse
    )
    rows, protos = _rows().to(dtype), _prototypes().to(dtype)
    rows_before, protos_before = rows.clone(), protos.clone()
    result, empty = _prepare(
        family,
        module,
        [rows, rows[:0]],
        torch.stack([protos, protos]),
        [_meta(), _meta()],
        smoothing=smoothing,
        threshold=0.37,
    )

    scores = torch.einsum("chw,nc->nhw", protos, rows[:, 6:])
    probability = family == "yolact" or (family == "yolov8" and smoothing)
    if probability:
        scores = 1 / (1 + torch.exp(-scores))
    boxes = torch.round(rows[:, :4] * 0.25)
    xx = torch.arange(16)[None, None, :]
    yy = torch.arange(8)[None, :, None]
    inside = (
        (xx >= boxes[:, 0, None, None])
        & (xx < boxes[:, 2, None, None])
        & (yy >= boxes[:, 1, None, None])
        & (yy < boxes[:, 3, None, None])
    )
    expected = scores * inside
    assert isinstance(result, InstanceDetections)
    assert isinstance(result.mask, InstanceSegmentationGrid)
    assert result.mask.scores.shape == (2, 8, 16)
    assert result.mask.scores.dtype == dtype
    assert torch.allclose(result.mask.scores, expected, atol=2 * torch.finfo(dtype).eps)
    assert result.mask.score_type == ("probabilities" if probability else "logits")
    assert result.mask.threshold == (
        0.37 if family == "yolov8" else 0.5 if family == "yolact" else 0
    )
    assert torch.equal(result.class_id, rows[:, 5].int())
    assert torch.equal(result.confidence, rows[:, 4])
    assert result.mask_size == result.image_size == result.mask_frame_size == (60, 80)
    assert torch.allclose(result.xyxy, rows[:, :4] / torch.tensor([0.8, 32 / 60] * 2))
    assert empty.mask.scores.shape == (0, 8, 16)
    assert empty.mask_size == result.mask_size
    assert empty.mask.grid_padding_ltrb == result.mask.grid_padding_ltrb
    assert torch.equal(rows, rows_before)
    assert torch.equal(protos, protos_before)
    protos.fill_(99)
    assert torch.allclose(result.mask.scores, expected, atol=2 * torch.finfo(dtype).eps)


@pytest.mark.parametrize("family", FAMILIES[:-1])
@pytest.mark.parametrize(
    "padding,crop",
    [
        ((4, 8, 4, 8), (0, 0, 80, 60)),
        ((-4, -4, -4, -4), (0, 0, 80, 60)),
        ((3, 5, 1, 3), (9, 7, 61, 43)),
        ((0, 0, 0, 0), (0, 0, 40, 30)),
    ],
)
def test_prototype_grid_keeps_native_geometry(family, padding, crop):
    module = import_module(f"inference_models.models.{family}.common")
    metadata = _meta(padding=padding, crop=crop)
    rows = _rows()
    result = _prepare(family, module, [rows], _prototypes()[None], [metadata])[0]
    assert result.mask.scores.shape == (2, 8, 16)
    assert result.mask.grid_padding_ltrb == tuple(round(value / 4) for value in padding)
    assert result.mask.image_crop_xywh == crop
    expected_boxes = (
        rows[:, :4] - torch.tensor([padding[0], padding[1]] * 2)
    ) / torch.tensor([metadata.scale_width, metadata.scale_height] * 2) + torch.tensor(
        [crop[0], crop[1]] * 2
    )
    assert torch.allclose(result.xyxy, expected_boxes)
    assert result.xyxy.dtype == torch.float32


@pytest.mark.parametrize("remap", [False, True])
@pytest.mark.parametrize("cap", [0, 2, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rfdetr_grid_selection_geometry_and_owned_scores(
    remap, cap, dtype, monkeypatch
):
    module = import_module("inference_models.models.rfdetr.common")
    monkeypatch.setattr(module, "align_instance_segmentation_results", _refuse)
    monkeypatch.setattr(
        module, "align_instance_segmentation_results_to_rle_masks", _refuse
    )
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.5, 0.5], [0.3, 0.4, 0.2, 0.3], [0.6, 0.6, 0.3, 0.2]]]
    )
    logits = torch.tensor([[[2.0, 3.0, -3.0], [4.0, -3.0, -4.0], [-4.0, -4.0, 5.0]]])
    masks = torch.stack([torch.full((6, 8), float(i), dtype=dtype) for i in range(3)])[
        None
    ]
    mapping = (
        ClassesReMapping(torch.tensor([1]), torch.tensor([-1, 0, -1]))
        if remap
        else None
    )
    meta = _meta(padding=(4, 6, 4, 6), crop=(7, 5, 63, 47), nonsquare=True)
    threshold = torch.tensor([0.6, 0.8])
    result = module.post_process_instance_segmentation_results_to_grid(
        bboxes=boxes,
        logits=logits,
        masks=masks,
        pre_processing_meta=[meta],
        threshold=threshold,
        num_classes=2,
        classes_re_mapping=mapping,
        max_detections=cap,
    )[0]
    # Independent scalar selection oracle: take one flattened top-k per query,
    # drop/remap classes, filter and retain the confidence-sorted cap.
    candidates = []
    scores, indices = logits.sigmoid().flatten().topk(3)
    for score, index in zip(scores.tolist(), indices.tolist()):
        query, class_id = divmod(index, 3)
        if class_id >= 2 or (remap and class_id != 1):
            continue
        class_id = 0 if remap else class_id
        if score > threshold[class_id]:
            candidates.append((score, query, class_id))
    candidates.sort(reverse=True)
    if cap is not None:
        candidates = candidates[:cap]
    assert len(result) == len(candidates)
    assert result.mask.scores.shape == (len(candidates), 6, 8)
    assert result.mask.scores.dtype == dtype
    assert result.mask.score_type == "logits" and result.mask.threshold == 0
    assert result.mask.grid_padding_ltrb == (0, 1, 0, 1)
    assert result.mask.image_crop_xywh == (7, 5, 63, 47)
    assert result.class_id.tolist() == [row[2] for row in candidates]
    assert torch.allclose(
        result.confidence, torch.tensor([row[0] for row in candidates])
    )
    for position, (_, query, _) in enumerate(candidates):
        assert torch.equal(result.mask.scores[position], masks[0, query])
    retained = result.mask.scores.clone()
    masks.fill_(99)
    assert torch.equal(result.mask.scores, retained)
    assert result.mask_size == result.image_size == result.mask_frame_size == (60, 80)


class _Stream:
    def __init__(self):
        self.actions = []

    def wait_event(self, event):
        self.actions.append("wait")

    def synchronize(self):
        self.actions.append("sync")


class _Event:
    def record(self, stream):
        stream.actions.append("consumed")


class _RawTensor(torch.Tensor):
    def record_stream(self, stream):
        stream.actions.append("record")


def _dispatch_method(path, namespace):
    """Compile actual dispatch bodies without importing optional GPU runtimes.

    This validates source dispatch logic, not TensorRT/CoreML runtime loading.
    Shared family functions are exercised through normal imports above.
    """
    tree = ast.parse(path.read_text())
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"post_process", "supported_mask_formats"}
    ]
    for method in methods:
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args + method.args.kwonlyargs:
            arg.annotation = None
        if method.args.kwarg:
            method.args.kwarg.annotation = None
        for node in ast.walk(method):
            if isinstance(node, ast.Name) and node.id.startswith("INFERENCE_MODELS_"):
                namespace[node.id] = 0.5
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["post_process"], namespace["supported_mask_formats"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("format_name", [None, "rle", "grid"])
def test_backend_dispatch_preserves_format_defaults_and_trt_order(
    backend, format_name, monkeypatch
):
    calls = []
    selected = [_rows()]
    sentinel = SimpleNamespace()
    stream = _Stream()
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)

    def prepare(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            stream.actions.append("prepare")
            return [sentinel]

        return run

    namespace = dict(
        torch=torch,
        ModelInputError=ModelInputError,
        ConfidenceFilter=lambda **_: SimpleNamespace(get_threshold=lambda _: 0.5),
        use_cuda_stream=lambda _: nullcontext(),
        run_yolov5_nms_for_instance_segmentation=lambda **_: selected,
        run_nms_for_instance_segmentation=lambda **_: selected,
        post_process_nms_fused_model_output=lambda **_: selected,
        decode_predicted_bboxes=lambda **kwargs: torch.zeros_like(kwargs["loc_data"]),
        get_trt_outputs_consumed_event=lambda _: None,
        get_deferred_postprocess_done_event=lambda _: None,
    )
    for name, fmt in [
        ("prepare_dense_masks", "dense"),
        ("prepare_rle_masks", "rle"),
        ("prepare_grid_masks", "grid"),
        ("post_process_instance_segmentation_results", "dense"),
        ("post_process_instance_segmentation_results_to_rle_masks", "rle"),
        ("post_process_instance_segmentation_results_to_grid", "grid"),
    ]:
        namespace[name] = prepare(fmt)
    post, formats = _dispatch_method(MODEL_ROOT / backend, namespace)
    model = SimpleNamespace(
        supported_mask_formats=formats(None),
        recommended_parameters=None,
        class_names=["a", "b", "c"],
        _post_process_stream=stream,
        _classes_re_mapping=None,
        _device="cpu",
        _inference_config=SimpleNamespace(post_processing=SimpleNamespace(fused=False)),
    )
    assert model.supported_mask_formats == {"dense", "rle", "grid"}
    raw = lambda *shape: torch.zeros(shape).as_subclass(_RawTensor)
    if backend.startswith("yolact/"):
        outputs = (
            raw(1, 2, 4),
            raw(1, 2, 4),
            raw(1, 2, 2),
            raw(1, 2, 4),
            raw(1, 8, 16, 2),
        )
    elif backend.startswith("rfdetr/"):
        outputs = (raw(1, 2, 4), raw(1, 2, 3), raw(1, 2, 8, 16))
        if backend.endswith("_pytorch.py"):
            outputs = dict(zip(("pred_boxes", "pred_logits", "pred_masks"), outputs))
    else:
        outputs = (raw(1, 2, 8), raw(1, 2, 8, 16))
    if backend.endswith("rfdetr_instance_segmentation_trt.py"):
        outputs[0]._trt_produce_event = _Event()
        outputs[0]._trt_graph_state = SimpleNamespace(consumer_done_event=None)
    kwargs = {"defer_postprocess_sync": True}
    if format_name is not None:
        kwargs["mask_format"] = format_name
    result = post(model, outputs, [_meta()], **kwargs)
    assert result == [sentinel]
    assert len(calls) == 1 and calls[0][0] == (format_name or "dense")
    assert ("masks_resolution_factor" in calls[0][1]) == (format_name != "grid")
    if backend.endswith("rfdetr_instance_segmentation_trt.py"):
        assert stream.actions[0] == "wait"
        assert (
            stream.actions.index("prepare")
            < stream.actions.index("consumed")
            < stream.actions.index("sync")
        )
    calls.clear()
    stream.actions.clear()
    with pytest.raises(ModelInputError, match="materialization"):
        post(model, None, None, mask_format="grid", masks_resolution_factor=0.5)
    assert not calls and not stream.actions
    with pytest.raises(ModelInputError):
        post(model, None, None, mask_format="unsupported")
    assert not calls and not stream.actions


@pytest.mark.parametrize(
    "backend", [name for name in BACKENDS if not name.endswith("_trt.py")]
)
def test_available_backend_imports_bind_real_grid_helper(backend):
    from inference_models.errors import MissingDependencyError

    module_name = "inference_models.models." + backend.removesuffix(".py").replace(
        "/", "."
    )
    try:
        module = import_module(module_name)
    except MissingDependencyError as error:
        pytest.skip(f"Optional backend dependency unavailable: {error}")
    except ModuleNotFoundError as error:
        if (
            backend != "rfdetr/rfdetr_instance_segmentation_pytorch.py"
            or error.name != "peft"
        ):
            raise
        pytest.skip("RF-DETR PyTorch optional peft dependency unavailable")
    family = backend.split("/")[0]
    helper = (
        "post_process_instance_segmentation_results_to_grid"
        if family == "rfdetr"
        else "prepare_grid_masks"
    )
    assert getattr(module, helper) is getattr(
        import_module(f"inference_models.models.{family}.common"), helper
    )
    tree = ast.parse((MODEL_ROOT / backend).read_text())
    name = next(node.name for node in tree.body if isinstance(node, ast.ClassDef))
    model_class = getattr(module, name)
    formats = model_class.supported_mask_formats.fget(None)
    with pytest.raises(ModelInputError, match="materialization"):
        model_class.post_process(
            SimpleNamespace(supported_mask_formats=formats),
            None,
            None,
            mask_format="grid",
            masks_resolution_factor=0.25,
        )


@pytest.mark.parametrize("suffix", ["onnx", "torch_script", "trt"])
def test_yolov11_inherits_the_grid_enabled_implementation(suffix):
    source = (MODEL_ROOT / "yolov11" / f"yolov11_{suffix}.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and "InstanceSegmentation" in node.name
    )
    assert any(
        "YOLOv8ForInstanceSegmentation" in ast.unparse(base) for base in cls.bases
    )
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name in {"post_process", "supported_mask_formats"}
        for node in cls.body
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_grid_helpers_preserve_an_empty_image_batch(family):
    module = import_module(f"inference_models.models.{family}.common")
    if family == "rfdetr":
        result = module.post_process_instance_segmentation_results_to_grid(
            bboxes=torch.empty(0, 2, 4),
            logits=torch.empty(0, 2, 3),
            masks=torch.empty(0, 2, 8, 16),
            pre_processing_meta=[],
            threshold=0.5,
            num_classes=3,
            classes_re_mapping=None,
        )
    else:
        result = _prepare(family, module, [], torch.empty(0, 2, 8, 16), [])
    assert result == []
