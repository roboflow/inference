"""Depth contract mechanics with synthetic backends, never real model proof."""

import ast
import hashlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from matplotlib import colormaps
from roboflow_workflows.v2.blocks.models.depth_estimation import v1 as depth
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.phases import PhaseFailure
from roboflow_workflows.v2.engine.reactions.testing import block_call

from tests.v2.support.paths import WORKFLOWS_ROOT


class FakeDepth:
    """Depth fixture with visible phase sizes and controllable bad outputs."""

    def __init__(self, *, global_flip=False, output=None):
        self.calls = []
        self.global_flip = global_flip
        self.output = output
        self.last_maps = []

    def pre_process(self, images, *, input_color_format):
        assert input_color_format == "rgb"
        self.calls.append(("pre", len(images)))
        return images, [tuple(im.shape[1:]) for im in images]

    def forward(self, images):
        self.calls.append(("forward", len(images)))
        maps = [im[0].to(torch.float32) for im in images]
        if self.global_flip:
            raw = torch.stack(maps).to(torch.bfloat16)
            # Exact left-associated expression in pinned DAV3 forward.
            maps = list((raw * -1) + raw.min() + raw.max())
        return maps

    def post_process(self, maps, metadata):
        self.calls.append(("post", len(metadata)))
        self.last_maps = [m.float() for m in maps] if self.global_flip else maps
        if self.output is not None:
            return self.output(self.last_maps)
        return self.last_maps


@pytest.fixture
def backend_types(monkeypatch):
    types = []
    for suffix, name, _, _ in depth._BACKENDS:
        module_name = f"inference_models.models.{suffix}"
        module = ModuleType(module_name)
        cls = type(name, (FakeDepth,), {"__module__": module_name})
        setattr(module, name, cls)
        monkeypatch.setitem(sys.modules, module_name, module)
        types.append(cls)
    return types


def make_image(values=(1, 2, 3, 4), *, size=(2, 2), image_id="test"):
    plane = torch.tensor(values, dtype=torch.uint8).reshape(size)
    return ImageData.from_tensor(plane.repeat(3, 1, 1), image_id=image_id)


def call(model, images, *, wanted=None, **kwargs):
    block = depth.DepthEstimationBlock(models={"anything": model})
    with block_call(block, wanted=wanted):
        result = block.run(image=Batch(images), model_id="anything", **kwargs)
    return result


@pytest.mark.parametrize("backend", range(5))
def test_direction_matches_latest_v1_tensor_formula(backend_types, backend):
    model = backend_types[backend]()
    image = make_image((1, 4, 2, 3))
    raw = image.tensor_image[0].float()
    proximity = -raw if backend >= 2 else raw
    expected = (proximity - proximity.min()) / (proximity.max() - proximity.min())
    result = call(model, [image])[0]
    v1 = _v1_tensor_result(proximity, image=image)
    assert torch.equal(result["normalized_depth"], expected)
    assert torch.equal(result["normalized_depth"], v1["normalized_depth"])
    np.testing.assert_array_equal(
        result["image"].tensor_image.permute(1, 2, 0).numpy(), v1["image"]
    )
    assert result["normalized_depth"].device == raw.device
    assert result["image"].tensor_image.dtype == torch.uint8
    assert result["image"].image_id == image.image_id
    assert result["image"].parent == image.parent
    assert result["image"].root == image.root


@pytest.mark.parametrize("backend", [0, 2, 3, 4])
def test_same_size_backends_keep_batches(backend_types, backend):
    model = backend_types[backend]()
    call(model, [make_image(), make_image()])
    assert model.calls == [("pre", 2), ("forward", 2), ("post", 2)]


def test_shape_chunks_scatter_cache_and_noncontiguous_rows(backend_types):
    loaded = []
    models = {}

    def load(request):
        loaded.append(request)
        models[request] = backend_types[0]()
        return models[request]

    block = depth.DepthEstimationBlock(models=ModelCache(load))
    images = [
        make_image((1, 4, 2, 3)),
        make_image(size=(1, 4)),
        make_image(),
        make_image((4, 1, 3, 2)),
        make_image(size=(1, 4)),
    ]
    for package in ("p", "p", "q"):
        with block_call(block):
            prepared = block.network_input(Batch(images), "local", package)
            assert prepared.request == ModelRequest("local", package)
            assert [chunk.rows for chunk in prepared.chunks] == [(0, 2, 3), (1, 4)]
            result = block.result(block.raw_output(prepared), prepared, Batch(images))
        for image, row in zip(images, result):
            raw = image.tensor_image[0].float()
            assert torch.equal(
                row["normalized_depth"], (raw - raw.min()) / (raw.max() - raw.min())
            )
            assert row["normalized_depth"].shape == image.size_hw
    assert loaded == [ModelRequest("local", "p"), ModelRequest("local", "q")]
    assert models[loaded[0]].calls.count(("forward", 3)) == 2
    assert models[loaded[0]].calls.count(("forward", 2)) == 2


def test_dav3_identity_ignores_request_spelling_and_packages(backend_types):
    class LocalSubclass(backend_types[1]):
        pass

    model = LocalSubclass()
    block = depth.DepthEstimationBlock(
        models={ModelRequest("yolo26-looking-id", "local-file"): model}
    )
    with block_call(block):
        result = block.run(
            image=Batch([make_image(), make_image()]),
            model_id="yolo26-looking-id",
            model_package_id="local-file",
        )
    assert len(result) == 2
    assert model.calls == [
        ("pre", 1),
        ("pre", 1),
        ("forward", 1),
        ("forward", 1),
        ("post", 1),
        ("post", 1),
    ]


def test_bf16_global_flip_counterexample_and_dav3_single_row_policy(backend_types):
    raw = torch.tensor([[[1, 2]], [[1024, 2048]]], dtype=torch.bfloat16)
    batched = (raw * -1) + raw.min() + raw.max()
    assert torch.equal(batched[0], torch.tensor([[2048, 2048]], dtype=torch.bfloat16))
    model = backend_types[1](global_flip=True)
    # Feed exact low precision values through a custom public preprocessor;
    # this arithmetic fixture is not real inference or measured CUDA behavior.
    source_preprocess = model.pre_process

    def preprocess(images, *, input_color_format):
        _, metadata = source_preprocess(images, input_color_format=input_color_format)
        value = raw[0] if int(images[0][0, 0, 0]) == 1 else raw[1]
        return [value.repeat(3, 1, 1)], metadata

    model.pre_process = preprocess
    result = call(
        model, [make_image((1, 2), size=(1, 2)), make_image((3, 4), size=(1, 2))]
    )
    assert [c for c in model.calls if c[0] == "forward"] == [
        ("forward", 1),
        ("forward", 1),
    ]
    assert torch.equal(result[0]["normalized_depth"], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(result[1]["normalized_depth"], torch.tensor([[1.0, 0.0]]))


def test_unknown_backend_fails_before_any_preprocessing(backend_types):
    unknown = FakeDepth()
    block = depth.DepthEstimationBlock(models={"depth-anything-v3/small": unknown})
    with pytest.raises(PhaseFailure, match="unsupported depth backend.*FakeDepth"):
        block.network_input(
            Batch([make_image(), make_image()]), "depth-anything-v3/small", None
        )
    assert unknown.calls == []


@pytest.mark.parametrize(
    "bad,match",
    [
        (lambda maps: [], "0 depth maps"),
        (lambda maps: torch.stack(maps), "expected a list"),
        (lambda maps: [torch.ones(1, 2, 2)], "floating depth map"),
        (lambda maps: [torch.ones(2, 3)], "floating depth map"),
        (lambda maps: [torch.ones(2, 2, dtype=torch.int32)], "floating depth map"),
        (lambda maps: [None], "floating depth map"),
        (lambda maps: [torch.ones(2, 2)], "no variation"),
        (lambda maps: [torch.tensor([[1.0, float("nan")], [2.0, 3.0]])], "nonfinite"),
        (lambda maps: [torch.tensor([[1.0, float("inf")], [2.0, 3.0]])], "nonfinite"),
        (lambda maps: [torch.tensor([[1.0, -float("inf")], [2.0, 3.0]])], "nonfinite"),
    ],
)
def test_bad_results_are_explicit(backend_types, bad, match):
    with pytest.raises(PhaseFailure, match=match):
        call(backend_types[0](output=bad), [make_image()])


def test_first_invalid_original_row_diagnostic(backend_types):
    # Rows 0 and 2 share a size chunk; its second map is the original row 2.
    model = backend_types[0](output=lambda maps: [maps[0], torch.ones_like(maps[1])])
    block = depth.DepthEstimationBlock(models={"a": model})
    with block_call(block), pytest.raises(
        PhaseFailure, match="row 2: Depth map has no variation"
    ):
        block.run(
            image=Batch([make_image(), make_image(size=(1, 4)), make_image()]),
            model_id="a",
        )


def test_all_viridis_entries_match_v1_rgb_and_endpoint_truncation():
    normalized = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 255
    image = ImageData.from_tensor(torch.zeros((3, 16, 16), dtype=torch.uint8))
    actual = (
        depth._visualization(image, normalized=normalized)
        .tensor_image.permute(1, 2, 0)
        .numpy()
    )
    indices = (normalized * 255).to(torch.uint8).numpy()
    expected = (colormaps["viridis"](indices)[:, :, :3] * 255).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)
    lut = depth._viridis_lut(torch.device("cpu"))
    np.testing.assert_array_equal(
        lut.numpy(),
        (colormaps["viridis"](np.arange(256))[:, :3] * 255).astype(np.uint8),
    )
    edges = torch.tensor([[0.0, 0.999, 1.0]])
    edge_image = make_image((1, 2, 3), size=(1, 3))
    viz = depth._visualization(edge_image, normalized=edges)
    assert torch.equal(viz.tensor_image[:, 0, :].T, lut[[0, 254, 255]])
    assert depth._viridis_lut(torch.device("cpu")) is lut


def test_no_mutation_or_alias_to_model_or_input(backend_types):
    model = backend_types[0]()
    image = make_image()
    before = image.tensor_image.clone()
    output = call(model, [image])[0]
    assert torch.equal(image.tensor_image, before)
    output["normalized_depth"].zero_()
    output["image"].tensor_image.zero_()
    assert torch.equal(image.tensor_image, before)
    assert torch.equal(model.last_maps[0], before[0].float())


def test_empty_batch_and_per_image_model_id_errors(backend_types):
    model = backend_types[0]()
    assert call(model, []) == []
    block = depth.DepthEstimationBlock(models={"a": model})
    with pytest.raises(PhaseFailure, match="model_id is one value for every image"):
        block.network_input(Batch([make_image(), make_image()]), Batch(["a"]), None)
    assert model.calls == []


def test_gray_rejected_before_model_calls(backend_types):
    model = backend_types[0]()
    image = ImageData.from_tensor(torch.ones((1, 2, 2), dtype=torch.uint8))
    with pytest.raises(PhaseFailure, match="3-channel RGB"):
        call(model, [image])
    assert model.calls == []


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("backend", range(5))
def test_finite_maps_match_pinned_v1_negate_then_normalize_exactly(
    backend_types, dtype, backend
):
    generator = torch.Generator().manual_seed(49)
    maps = (
        torch.rand((7, 11), generator=generator, dtype=torch.float64).to(dtype) * 32
        - 16
    )
    model = backend_types[backend](output=lambda _: [maps])
    image = make_image(tuple(range(77)), size=(7, 11))
    actual = call(model, [image], wanted=["normalized_depth"])[0]["normalized_depth"]
    v1_map = -maps if backend >= 2 else maps
    expected = (v1_map - v1_map.min()) / (v1_map.max() - v1_map.min())
    assert torch.equal(actual, expected)
    assert actual.dtype == maps.dtype
    assert actual.device == maps.device


def test_class_name_alone_is_not_backend_identity(backend_types):
    spoof = type("DepthAnythingV3Torch", (FakeDepth,), {})()
    with pytest.raises(PhaseFailure, match="unsupported depth backend"):
        call(spoof, [make_image()])


def test_depth_only_demand_avoids_even_color_table_import(backend_types, monkeypatch):
    def unwanted(*args, **kwargs):
        raise AssertionError("visualization must not run")

    monkeypatch.setattr(depth, "_viridis_lut", unwanted)
    result = call(backend_types[0](), [make_image()], wanted=["normalized_depth"])
    assert set(result[0]) == {"normalized_depth"}


def _v1_tensor_result(proximity, *, image):
    # Execute the pinned V1 method and renderer unchanged, without importing
    # its legacy bootstrap, manager dependencies or hosted client. Only the
    # public manager/image boundary is replaced by an offline fixture.
    source = (
        WORKFLOWS_ROOT
        / "roboflow_workflows/core_steps/models/foundation/depth_estimation/v1_tensor.py"
    )
    content = source.read_bytes()
    assert (
        hashlib.sha256(content).hexdigest()
        == "b2da783804348d66cb38a24dbe1e477e96881434cbda74d8a276b90c6bbff056"
    )
    tree = ast.parse(content)
    renderer = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_depth_to_visualization"
    )
    block = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DepthEstimationBlockV1"
    )
    method = next(
        n
        for n in block.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_locally"
    )
    wrapper = ast.ClassDef(
        name="PinnedV1", bases=[], keywords=[], body=[method], decorator_list=[]
    )
    subset = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            renderer,
            wrapper,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "np": np,
        "plt": SimpleNamespace(get_cmap=lambda name: colormaps[name]),
        "WORKFLOWS_IMAGE_TENSOR_DEVICE": "cpu",
        "WorkflowImageData": SimpleNamespace(
            copy_and_replace=lambda **kw: kw["numpy_image"]
        ),
    }
    exec(compile(ast.fix_missing_locations(subset), str(source), "exec"), namespace)
    model = namespace["PinnedV1"]()
    model._api_key = None
    model._model_manager = SimpleNamespace(
        add_model=lambda **kw: None,
        run_tensor_native_inference=lambda *args, **kw: [proximity],
    )
    wrapped_image = SimpleNamespace(
        is_tensor_materialised=lambda: True, tensor_image=image.tensor_image
    )
    result = model.run_locally(images=[wrapped_image], model_version="fixture")[0]
    return result
