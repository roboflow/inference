"""Tests of ``ImageData``, the tensor image carrier of the native V2 catalogue.

Geometry expectations are computed by hand from the one published equation::

    frame_xy = local_xy * scale_xy + offset_xy

and, for the V1 bridge, from V1's ``root_xy = local_xy / scaling + offset``.
V1 parity cases use the real V1 ``WorkflowImageData``, its crop constructors,
``build_native_image_metadata`` and ``serialise_image`` as oracles. Device
cases run on every available device; this machine may lack CUDA.
"""

import base64
import dataclasses
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.common.serializers import serialise_image
from roboflow_workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.blocks.image_data import (
    CompositeSource,
    FrameMapping,
    ImageData,
)
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import ContractError

DEVICES = [
    "cpu",
    *(["mps"] if torch.backends.mps.is_available() else []),
    *(["cuda"] if torch.cuda.is_available() else []),
]
V1_KEYS = (
    "parent_id",
    "root_parent_id",
    "image_dimensions",
    "parent_dimensions",
    "root_parent_dimensions",
    "parent_coordinates",
    "root_parent_coordinates",
)


def _coordinate_image(height: int, width: int, *, device: str = "cpu") -> ImageData:
    """Root whose red channel holds x and green channel holds y of each pixel."""
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    pixels = torch.stack([xs, ys, torch.zeros_like(xs)]).to(torch.uint8)

    return ImageData.from_tensor(pixels.to(device), image_id="root")


def _random_bgr(height: int, width: int, *, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    bgr = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    return bgr


def _video_metadata() -> VideoMetadata:
    metadata = VideoMetadata(
        video_identifier="camera-7",
        frame_number=42,
        frame_timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
        fps=30.0,
    )

    return metadata


def _v1_restore(metadata: dict, point: tuple) -> tuple:
    # V1 convention: root_xy = local_xy / scaling + root_parent_coordinates.
    scaling = metadata["scaling_relative_to_root_parent"]
    scale_x, scale_y = scaling if isinstance(scaling, list) else (scaling, scaling)
    offset_x, offset_y = metadata["root_parent_coordinates"]
    restored = (point[0] / scale_x + offset_x, point[1] / scale_y + offset_y)

    return restored


MALFORMED_COMPOSITE_FIELDS = [
    pytest.param("index", (1.9,), ValueError, "index", id="index-float"),
    pytest.param("index", (1.0,), ValueError, "index", id="index-integral-float"),
    pytest.param("index", (True,), ValueError, "index", id="index-bool"),
    pytest.param("index", (0, -2), ValueError, "index", id="index-negative"),
    pytest.param("index", (), ValueError, "index", id="index-empty"),
    pytest.param("index", "0", ValueError, "index", id="index-string"),
    pytest.param("index", 3, ValueError, "index", id="index-scalar"),
    pytest.param("image_id", "", ValueError, "image_id", id="image-id-empty"),
    pytest.param("image_id", 7, ValueError, "image_id", id="image-id-number"),
    pytest.param("image_id", None, ValueError, "image_id", id="image-id-none"),
    pytest.param("canvas_xyxy", (5, 2, 1), ValueError, "four integers", id="rect-3"),
    pytest.param(
        "canvas_xyxy", (0, 0, 4, 4, 4), ValueError, "four integers", id="rect-5"
    ),
    pytest.param(
        "canvas_xyxy", (0, 0, 4.5, 4), ValueError, "four integers", id="rect-float"
    ),
    pytest.param(
        "canvas_xyxy", (0, 0, True, 4), ValueError, "four integers", id="rect-bool"
    ),
    pytest.param(
        "canvas_xyxy", (4, 0, 4, 4), ValueError, "x0 < x1", id="rect-no-width"
    ),
    pytest.param(
        "canvas_xyxy", (0, 4, 4, 4), ValueError, "y0 < y1", id="rect-no-height"
    ),
    pytest.param(
        "canvas_xyxy", (5, 2, 1, 3), ValueError, "x0 < x1", id="rect-inverted"
    ),
    pytest.param(
        "parent", {"frame_id": "source"}, TypeError, "FrameMapping", id="parent-dict"
    ),
    pytest.param("root", None, TypeError, "FrameMapping", id="root-none"),
]
# Fields that are plain JSON values on the wire; malformed mappings are
# covered separately because the wire always holds dicts there.
MALFORMED_WIRE_FIELDS = [
    case
    for case in MALFORMED_COMPOSITE_FIELDS
    if case.values[0] in ("index", "image_id", "canvas_xyxy")
]


@pytest.fixture
def host_transfers_forbidden(monkeypatch):
    """Fail on any host copy or device move of a tensor."""

    def forbidden(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"unexpected Tensor.{name} call")

        return fail

    for name in ("cpu", "numpy", "tolist", "item", "__array__", "cuda"):
        monkeypatch.setattr(torch.Tensor, name, forbidden(name))

    original_to = torch.Tensor.to

    def to_without_device(self, *args, **kwargs):
        targets = list(args) + [kwargs.get("device")]
        if any(isinstance(target, (str, torch.device)) for target in targets):
            raise AssertionError(f"unexpected device move Tensor.to{args}")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_without_device)


class TestFrameMapping:
    def test_maps_and_composes_local_to_target(self) -> None:
        inner = FrameMapping("mid", (50, 60), scale_xy=(2.0, 0.5), offset_xy=(3, 4))
        outer = FrameMapping("root", (500, 600), scale_xy=(10, 3), offset_xy=(-1, 7))

        composed = inner.then(outer)

        assert inner.map_xy(5, 8) == (13.0, 8.0)
        assert composed.map_xy(5, 8) == outer.map_xy(*inner.map_xy(5, 8))
        assert composed == FrameMapping(
            "root", (500, 600), scale_xy=(20.0, 1.5), offset_xy=(29.0, 19.0)
        )

    def test_dictionary_round_trip(self) -> None:
        mapping = FrameMapping("f", (3, 4), scale_xy=(0.25, 3.0), offset_xy=(-2.5, 1))

        assert FrameMapping.from_dict(mapping.to_dict()) == mapping
        assert FrameMapping.from_dict(json.loads(json.dumps(mapping.to_dict()))) == (
            mapping
        )

    @pytest.mark.parametrize(
        "arguments, fragment",
        [
            ({"frame_id": "", "frame_size_hw": (1, 1)}, "frame_id"),
            ({"frame_id": "f", "frame_size_hw": (0, 4)}, "Size"),
            ({"frame_id": "f", "frame_size_hw": (4,)}, "Size"),
            ({"frame_id": "f", "frame_size_hw": (True, 4)}, "Size"),
            ({"frame_id": "f", "frame_size_hw": (1, 1), "scale_xy": (0, 1)}, "scale"),
            ({"frame_id": "f", "frame_size_hw": (1, 1), "scale_xy": (-1, 1)}, "scale"),
            ({"frame_id": "f", "frame_size_hw": (1, 1), "scale_xy": 2.0}, "scale"),
            (
                {"frame_id": "f", "frame_size_hw": (1, 1), "offset_xy": (0, np.nan)},
                "offset",
            ),
            (
                {"frame_id": "f", "frame_size_hw": (1, 1), "offset_xy": (np.inf, 0)},
                "offset",
            ),
        ],
    )
    def test_rejects_malformed_values(self, arguments, fragment) -> None:
        with pytest.raises(ValueError, match=fragment):
            FrameMapping(**arguments)

    def test_from_dict_names_a_missing_key(self) -> None:
        with pytest.raises(ValueError, match="scale_xy"):
            FrameMapping.from_dict({"frame_id": "f", "size_hw": [1, 1]})

    def test_records_are_immutable(self) -> None:
        mapping = FrameMapping.identity("f", (2, 3))
        image = ImageData.from_tensor(torch.zeros((3, 2, 3), dtype=torch.uint8))

        with pytest.raises(dataclasses.FrozenInstanceError):
            mapping.offset_xy = (1.0, 1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            image.root = mapping


class TestComposition:
    def test_crop_resize_crop_matches_hand_computed_root_transform(self) -> None:
        root = ImageData.from_tensor(
            torch.zeros((3, 100, 100), dtype=torch.uint8), image_id="root"
        )

        first = root.crop((10, 20, 50, 60))
        doubled = first.resize((80, 80))
        second = doubled.crop((3, 5, 40, 40))

        assert second.root == FrameMapping(
            "root", (100, 100), scale_xy=(0.5, 0.5), offset_xy=(11.5, 22.5)
        )
        assert second.parent == FrameMapping(
            doubled.image_id, (80, 80), offset_xy=(3, 5)
        )

    def test_resize_then_crop_differs_from_crop_then_resize(self) -> None:
        root = ImageData.from_tensor(
            torch.zeros((3, 10, 10), dtype=torch.uint8), image_id="root"
        )

        resized_then_cropped = root.resize((40, 20)).crop((6, 12, 20, 40))
        cropped_then_resized = root.crop((6, 12, 10, 10))

        assert resized_then_cropped.root.map_xy(2, 4) == (4.0, 4.0)
        assert cropped_then_resized is None

    def test_anisotropic_nested_case_restores_asymmetric_root_box(self) -> None:
        root = ImageData.from_tensor(
            torch.zeros((3, 300, 400), dtype=torch.uint8), image_id="root"
        )

        first = root.crop((30, 70, 230, 170))
        resized = first.resize((25, 100))
        second = resized.crop((5, 3, 50, 20))

        assert resized.parent.scale_xy == (2.0, 4.0)
        assert second.root.map_xy(2, 4) == (44.0, 98.0)
        assert second.root.map_xy(6, 8) == (52.0, 114.0)
        metadata = second.prediction_metadata()
        assert _v1_restore(metadata, (2, 4)) == (44.0, 98.0)
        assert _v1_restore(metadata, (6, 8)) == (52.0, 114.0)

    def test_resize_ratio_uses_actual_integer_output_size(self) -> None:
        image = ImageData.from_tensor(torch.zeros((3, 10, 7), dtype=torch.uint8))

        resized = image.resize((3, 3))

        assert resized.size_hw == (3, 3)
        assert resized.parent.scale_xy == (7 / 3, 10 / 3)

    def test_origin_zero_crop_is_not_the_root(self) -> None:
        root = ImageData.from_tensor(
            torch.zeros((3, 80, 100), dtype=torch.uint8), image_id="root"
        )

        crop = root.crop((0, 0, 50, 40))
        metadata = crop.prediction_metadata()

        assert crop.image_id != "root"
        assert metadata["parent_id"] == crop.image_id
        assert metadata["parent_frame_id"] == metadata["root_parent_id"] == "root"
        assert metadata["image_dimensions"] == [40, 50]
        assert metadata["root_parent_dimensions"] == [80, 100]
        assert metadata["root_parent_coordinates"] == [0.0, 0.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_pixels_agree_with_provenance_through_crop_resize_crop(
        self, device
    ) -> None:
        root = _coordinate_image(120, 200, device=device)

        first = root.crop((30, 20, 190, 100))
        resized = first.resize((40, 40), interpolation="nearest")
        second = resized.crop((5, 3, 25, 23))

        assert resized.parent.scale_xy == (4.0, 2.0)
        pixels = second.tensor_image.to("cpu")
        for y in range(second.height):
            for x in range(second.width):
                root_x, root_y = second.root.map_xy(x, y)
                assert (int(pixels[0, y, x]), int(pixels[1, y, x])) == (
                    root_x,
                    root_y,
                )


class TestPredictionMetadata:
    def test_v1_keys_with_reciprocal_scales(self) -> None:
        root = ImageData.from_tensor(
            torch.zeros((3, 40, 80), dtype=torch.uint8), image_id="root"
        )
        crop = root.crop((8, 4, 48, 24))
        resized = crop.resize((5, 10))

        metadata = resized.prediction_metadata()

        assert metadata == {
            "parent_id": resized.image_id,
            "parent_frame_id": crop.image_id,
            "root_parent_id": "root",
            "image_dimensions": [5, 10],
            "parent_dimensions": [20, 40],
            "root_parent_dimensions": [40, 80],
            "parent_coordinates": [0.0, 0.0],
            "root_parent_coordinates": [8.0, 4.0],
            "scaling_relative_to_parent": 0.25,
            "scaling_relative_to_root_parent": 0.25,
        }

    def test_anisotropic_scale_is_an_xy_pair(self) -> None:
        image = ImageData.from_tensor(torch.zeros((3, 10, 20), dtype=torch.uint8))

        metadata = image.resize((5, 5)).prediction_metadata()

        assert metadata["scaling_relative_to_parent"] == [0.25, 0.5]

    def test_each_call_returns_independent_plain_values(self) -> None:
        canvas = ImageData.composite(
            torch.zeros((3, 4, 4), dtype=torch.uint8),
            sources=[
                CompositeSource.place(
                    ImageData.from_tensor(torch.zeros((3, 2, 2), dtype=torch.uint8)),
                    index=(0,),
                    canvas_xyxy=(0, 0, 4, 4),
                )
            ],
        )
        first = canvas.prediction_metadata()

        first["parent_dimensions"].append(5)
        first["composite_sources"][0]["parent"]["offset_xy"][0] = 99.0
        second = canvas.prediction_metadata()

        assert second["parent_dimensions"] == [4, 4]
        assert second["composite_sources"][0]["parent"]["offset_xy"] == [0.0, 0.0]
        assert canvas.composite_sources[0].parent.offset_xy == (0.0, 0.0)
        assert json.loads(json.dumps(second, allow_nan=False)) == second

    def test_ordinary_images_carry_no_composite_keys(self) -> None:
        image = ImageData.from_tensor(torch.zeros((3, 2, 2), dtype=torch.uint8))

        metadata = image.prediction_metadata()

        assert "is_composite" not in metadata
        assert "composite_sources" not in metadata


class TestConstructionAndIngress:
    @pytest.mark.parametrize("device", DEVICES)
    def test_tensor_input_is_neither_copied_nor_moved(self, device) -> None:
        pixels = torch.zeros((3, 6, 8), dtype=torch.uint8, device=device)

        image = IMAGE_KIND.to_payload(pixels)

        assert image.tensor_image is pixels
        assert image.device == pixels.device
        assert (
            image.parent == image.root == FrameMapping.identity(image.image_id, (6, 8))
        )

    def test_non_contiguous_tensor_is_kept_as_given(self) -> None:
        pixels = torch.zeros((3, 8, 6), dtype=torch.uint8).transpose(1, 2)

        image = ImageData.from_tensor(pixels)

        assert image.tensor_image is pixels
        assert image.size_hw == (6, 8)

    def test_raw_inputs_get_distinct_ids_and_given_ids_are_kept(self) -> None:
        pixels = torch.zeros((3, 2, 2), dtype=torch.uint8)

        first, second = IMAGE_KIND.to_payload(pixels), IMAGE_KIND.to_payload(pixels)
        named = ImageData.from_tensor(pixels, image_id="camera-a/frame-7")

        assert first.image_id != second.image_id
        assert first.image_id.startswith("input-")
        assert named.image_id == named.root.frame_id == "camera-a/frame-7"
        assert IMAGE_KIND.to_payload(named) is named

    @pytest.mark.parametrize(
        "pixels, error, fragment",
        [
            (torch.zeros((4, 5, 3), dtype=torch.uint8), ValueError, "permute"),
            (torch.zeros((4, 5, 6), dtype=torch.uint8), ValueError, "channels"),
            (torch.zeros((5, 6), dtype=torch.uint8), ValueError, "channels"),
            (torch.zeros((3, 5, 6), dtype=torch.float32), ValueError, "uint8"),
            (torch.zeros((3, 0, 6), dtype=torch.uint8), ValueError, "empty"),
            (np.zeros((3, 5, 6), dtype=np.uint8), TypeError, "torch.Tensor"),
        ],
    )
    def test_rejects_ambiguous_or_invalid_pixels(self, pixels, error, fragment) -> None:
        with pytest.raises(error, match=fragment):
            ImageData.from_tensor(pixels)

    def test_three_by_w_by_three_is_read_as_channels_first(self) -> None:
        image = ImageData.from_tensor(torch.zeros((3, 7, 3), dtype=torch.uint8))

        assert image.size_hw == (7, 3)

    def test_numpy_rgb_input_keeps_channel_order_and_is_copied(self) -> None:
        array = np.zeros((2, 3, 3), dtype=np.uint8)
        array[..., 0], array[..., 1], array[..., 2] = 10, 20, 30

        image = IMAGE_KIND.to_payload(array)
        array[...] = 0

        assert image.tensor_image[:, 1, 2].tolist() == [10, 20, 30]
        assert image.device == torch.device("cpu")

    def test_numpy_single_pixel_is_copied_not_viewed(self) -> None:
        array = np.array([[[1, 2, 3]]], dtype=np.uint8)

        image = ImageData.from_numpy_rgb(array)
        array[...] = 0

        assert image.tensor_image.flatten().tolist() == [1, 2, 3]

    def test_numpy_grayscale_becomes_one_channel(self) -> None:
        image = ImageData.from_numpy_rgb(np.full((4, 5), 7, dtype=np.uint8))

        assert (image.channels, image.size_hw) == (1, (4, 5))

    @pytest.mark.parametrize(
        "array",
        [
            np.zeros((4, 5, 4), dtype=np.uint8),
            np.zeros((4, 5, 3), dtype=np.float32),
            np.zeros((3, 4, 5, 3), dtype=np.uint8),
        ],
    )
    def test_rejects_other_numpy_layouts(self, array) -> None:
        with pytest.raises(ValueError, match="Expected a"):
            IMAGE_KIND.to_payload(array)

    def test_unknown_input_type_lists_accepted_forms(self) -> None:
        with pytest.raises(ContractError, match="V1 WorkflowImageData"):
            IMAGE_KIND.to_payload(3.5)

    def test_invalid_records_are_rejected(self) -> None:
        pixels = torch.zeros((3, 2, 2), dtype=torch.uint8)
        frame = FrameMapping.identity("x", (2, 2))

        with pytest.raises(ValueError, match="image_id"):
            ImageData(pixels, image_id="", parent=frame, root=frame)
        with pytest.raises(TypeError, match="parent must be a FrameMapping"):
            ImageData(pixels, image_id="x", parent={"frame_id": "x"}, root=frame)
        with pytest.raises(TypeError, match="VideoMetadata"):
            ImageData(pixels, image_id="x", parent=frame, root=frame, video_metadata={})
        with pytest.raises(ValueError, match="non-empty string"):
            ImageData.from_tensor(pixels, image_id="")


class TestOperations:
    @pytest.mark.parametrize("device", DEVICES)
    def test_crop_is_an_independent_copy_on_the_same_device(self, device) -> None:
        image = _coordinate_image(10, 12, device=device)
        original = image.tensor_image.clone()

        crop = image.crop((2, 3, 9, 8))
        crop.tensor_image[...] = 255

        assert crop.device == image.device
        assert crop.tensor_image.is_contiguous()
        assert torch.equal(image.tensor_image, original)
        assert (
            crop.tensor_image.untyped_storage().data_ptr()
            != image.tensor_image.untyped_storage().data_ptr()
        )

    def test_crop_clips_and_returns_none_when_nothing_remains(self) -> None:
        image = _coordinate_image(10, 12)

        clipped = image.crop((-4, -2, 5, 30))

        assert clipped.size_hw == (10, 5)
        assert clipped.parent.offset_xy == (0.0, 0.0)
        assert image.crop((12, 0, 20, 5)) is None
        assert image.crop((3, 3, 3, 9)) is None
        with pytest.raises(ValueError, match="four integers"):
            image.crop((0.5, 0, 4, 4))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
    def test_resize_stays_on_device_and_keeps_constant_images(
        self, device, interpolation
    ) -> None:
        image = ImageData.from_tensor(
            torch.full((3, 9, 13), 77, dtype=torch.uint8, device=device)
        )

        resized = image.resize((4, 5), interpolation=interpolation)

        assert resized.device == image.device
        assert resized.tensor_image.dtype == torch.uint8
        assert tuple(resized.tensor_image.shape) == (3, 4, 5)
        assert bool((resized.tensor_image == 77).all())

    def test_resize_rejects_unknown_interpolation_and_bad_size(self) -> None:
        image = _coordinate_image(4, 4)

        with pytest.raises(ValueError, match="interpolation"):
            image.resize((2, 2), interpolation="cubic")
        with pytest.raises(ValueError, match="positive"):
            image.resize((0, 2))

    def test_with_pixels_keeps_identity_geometry_and_metadata(self) -> None:
        image = ImageData.from_tensor(
            torch.zeros((3, 4, 5), dtype=torch.uint8), video_metadata=_video_metadata()
        ).crop((1, 1, 4, 4))
        gray = torch.full((1, 3, 3), 9, dtype=torch.uint8)

        updated = image.with_pixels(gray)

        assert updated.tensor_image is gray
        assert (updated.image_id, updated.parent, updated.root) == (
            image.image_id,
            image.parent,
            image.root,
        )
        assert updated.video_metadata is image.video_metadata
        assert updated.channels == 1
        with pytest.raises(ValueError, match="keeps geometry"):
            image.with_pixels(torch.zeros((3, 4, 3), dtype=torch.uint8))

    def test_derived_images_keep_video_metadata(self) -> None:
        video_metadata = _video_metadata()
        image = ImageData.from_tensor(
            torch.zeros((3, 8, 8), dtype=torch.uint8), video_metadata=video_metadata
        )

        derived = image.crop((0, 0, 4, 4)).resize((2, 2))

        assert derived.video_metadata is video_metadata

    def test_host_transfer_guard_detects_transfers(
        self, host_transfers_forbidden
    ) -> None:
        pixels = torch.zeros((3, 2, 2), dtype=torch.uint8)

        for transfer in (
            pixels.cpu,
            pixels.numpy,
            lambda: pixels.to("cpu"),
            lambda: np.asarray(pixels),
        ):
            with pytest.raises(AssertionError, match="unexpected"):
                transfer()
        assert pixels.to(torch.float32).dtype == torch.float32

    @pytest.mark.parametrize("device", DEVICES)
    def test_operations_make_no_host_copies_or_device_moves(
        self, device, host_transfers_forbidden
    ) -> None:
        image = ImageData.from_tensor(
            torch.zeros((3, 20, 30), dtype=torch.uint8, device=device)
        )

        crop = image.crop((3, 2, 25, 18))
        nearest = crop.resize((8, 11), interpolation="nearest")
        bilinear = crop.resize((7, 5))
        inverted = bilinear.with_pixels(255 - bilinear.tensor_image)
        canvas = ImageData.composite(
            torch.zeros((3, 8, 16), dtype=torch.uint8, device=device),
            sources=[
                CompositeSource.place(nearest, index=(0,), canvas_xyxy=(0, 0, 8, 8))
            ],
        )
        canvas.crop((1, 1, 5, 5)).prediction_metadata()

        for result in (crop, nearest, bilinear, inverted, canvas):
            assert result.device == image.device


class TestComposite:
    def test_placement_maps_canvas_points_to_source_and_its_root(self) -> None:
        root = ImageData.from_tensor(
            torch.zeros((3, 100, 120), dtype=torch.uint8), image_id="root"
        )
        source = root.crop((30, 20, 90, 60))

        placement = CompositeSource.place(
            source, index=(0, 2), canvas_xyxy=(8, 0, 16, 8)
        )

        assert placement.parent.frame_id == source.image_id
        assert placement.parent.map_xy(8, 0) == (0.0, 0.0)
        assert placement.parent.map_xy(16, 8) == (60.0, 40.0)
        assert placement.root.frame_id == "root"
        assert placement.root.map_xy(8, 0) == (30.0, 20.0)
        assert placement.root.map_xy(16, 8) == (90.0, 60.0)
        assert CompositeSource.from_dict(placement.to_dict()) == placement

    def test_composite_is_its_own_root_and_crops_inherit_the_marker(self) -> None:
        source = ImageData.from_tensor(torch.zeros((3, 4, 4), dtype=torch.uint8))
        placement = CompositeSource.place(source, index=(0,), canvas_xyxy=(0, 0, 4, 4))

        canvas = ImageData.composite(
            torch.zeros((3, 4, 8), dtype=torch.uint8), sources=[placement]
        )
        crop = canvas.crop((1, 1, 6, 3))

        assert canvas.image_id.startswith("mosaic-")
        assert canvas.root == FrameMapping.identity(canvas.image_id, (4, 8))
        assert canvas.is_composite and crop.is_composite
        assert crop.composite_sources == (placement,)
        assert crop.root.frame_id == canvas.image_id
        metadata = crop.prediction_metadata()
        assert metadata["is_composite"] is True
        assert metadata["composite_sources"] == [placement.to_dict()]

    def test_empty_composite_is_still_composite(self) -> None:
        canvas = ImageData.composite(
            torch.zeros((3, 2, 2), dtype=torch.uint8), sources=()
        )

        assert canvas.is_composite
        assert canvas.prediction_metadata()["composite_sources"] == []

    def test_placement_rejects_an_empty_rectangle(self) -> None:
        source = ImageData.from_tensor(torch.zeros((3, 4, 4), dtype=torch.uint8))

        with pytest.raises(ValueError, match="x0 < x1"):
            CompositeSource.place(source, index=(0,), canvas_xyxy=(4, 0, 4, 4))

    @pytest.mark.parametrize(
        "field, value, error, fragment", MALFORMED_COMPOSITE_FIELDS
    )
    def test_construction_rejects_each_malformed_field(
        self, field, value, error, fragment
    ) -> None:
        fields = {**_valid_composite_fields(), field: value}

        with pytest.raises(error, match=fragment):
            CompositeSource(**fields)

    def test_valid_fields_are_stored_as_plain_tuples(self) -> None:
        fields = {
            **_valid_composite_fields(),
            "index": [np.int64(2), 0],
            "canvas_xyxy": [0, 0, np.int32(4), 4],
        }

        placement = CompositeSource(**fields)

        assert placement.index == (2, 0)
        assert placement.canvas_xyxy == (0, 0, 4, 4)
        assert json.loads(json.dumps(placement.to_dict())) == placement.to_dict()


def _valid_composite_fields() -> dict:
    fields = {
        "index": (1, 0),
        "image_id": "source",
        "canvas_xyxy": (0, 0, 4, 4),
        "parent": FrameMapping("source", (8, 8), scale_xy=(2.0, 2.0)),
        "root": FrameMapping("root", (8, 8), scale_xy=(2.0, 2.0)),
    }

    return fields


def _composite_wire() -> dict:
    source = ImageData.from_tensor(torch.zeros((3, 8, 8), dtype=torch.uint8))
    canvas = ImageData.composite(
        torch.zeros((3, 4, 4), dtype=torch.uint8),
        sources=[CompositeSource.place(source, index=(1, 0), canvas_xyxy=(0, 0, 4, 4))],
    )
    wire = json.loads(json.dumps(IMAGE_KIND.to_serialized(canvas)))

    return wire


class TestMalformedCompositeWire:
    """Encoded composite provenance is rejected, never coerced, at ingress."""

    def test_valid_wire_is_accepted(self) -> None:
        image = IMAGE_KIND.to_payload(_composite_wire())

        assert image.composite_sources[0].index == (1, 0)

    @pytest.mark.parametrize("field, value, error, fragment", MALFORMED_WIRE_FIELDS)
    def test_each_malformed_field_is_rejected(
        self, field, value, error, fragment
    ) -> None:
        wire = _composite_wire()
        wire["composite_sources"][0][field] = value

        with pytest.raises(ValueError, match=fragment):
            IMAGE_KIND.to_payload(wire)

    @pytest.mark.parametrize(
        "mapping",
        [
            "not a mapping",
            {"frame_id": "source", "size_hw": [8, 8], "scale_xy": [0, 1]},
            {"frame_id": "source", "size_hw": [8, 8], "offset_xy": [0, 0]},
        ],
        ids=["string", "zero-scale", "missing-scale"],
    )
    @pytest.mark.parametrize("field", ["parent", "root"])
    def test_malformed_mappings_are_rejected(self, field, mapping) -> None:
        wire = _composite_wire()
        wire["composite_sources"][0][field] = mapping

        with pytest.raises(ValueError, match="frame mapping|scale_xy"):
            IMAGE_KIND.to_payload(wire)

    @pytest.mark.parametrize("missing", ["index", "image_id", "canvas_xyxy"])
    def test_missing_fields_are_rejected(self, missing) -> None:
        wire = _composite_wire()
        del wire["composite_sources"][0][missing]

        with pytest.raises(ValueError, match="Invalid composite source"):
            IMAGE_KIND.to_payload(wire)

    def test_reviewed_combined_counterexample_is_rejected(self) -> None:
        wire = _composite_wire()
        wire["composite_sources"][0].update(
            index=[1.9, -2, True], canvas_xyxy=[5, 2, 1], image_id=""
        )

        with pytest.raises(ValueError, match="index"):
            IMAGE_KIND.to_payload(wire)


class TestSerializedRoundTrip:
    def test_provenance_ids_scales_and_pixels_survive_strict_json(self) -> None:
        root = ImageData.from_tensor(
            torch.from_numpy(_random_bgr(40, 60).transpose(2, 0, 1).copy()),
            image_id="root",
            video_metadata=_video_metadata(),
        )
        image = root.crop((7, 3, 50, 33)).resize((10, 20)).crop((2, 1, 15, 9))

        wire = json.loads(json.dumps(IMAGE_KIND.to_serialized(image), allow_nan=False))
        restored = IMAGE_KIND.to_payload(wire)

        assert torch.equal(restored.tensor_image, image.tensor_image)
        assert (restored.image_id, restored.parent, restored.root) == (
            image.image_id,
            image.parent,
            image.root,
        )
        assert restored.root.scale_xy == (43 / 20, 30 / 10)
        assert restored.video_metadata == image.video_metadata
        assert restored.prediction_metadata() == image.prediction_metadata()
        assert restored.composite_sources is None

    def test_composite_and_grayscale_survive(self) -> None:
        source = ImageData.from_tensor(torch.zeros((3, 4, 6), dtype=torch.uint8))
        canvas = ImageData.composite(
            torch.arange(24, dtype=torch.uint8).reshape(1, 4, 6),
            sources=[
                CompositeSource.place(source, index=(3,), canvas_xyxy=(0, 0, 6, 4))
            ],
        )

        restored = IMAGE_KIND.to_payload(IMAGE_KIND.to_serialized(canvas))

        assert restored.channels == 1
        assert torch.equal(restored.tensor_image, canvas.tensor_image)
        assert restored.composite_sources == canvas.composite_sources
        assert restored.is_composite

    @pytest.mark.parametrize("device", DEVICES)
    def test_serialization_reads_device_pixels_without_changing_them(
        self, device
    ) -> None:
        pixels = torch.randint(0, 256, (3, 5, 4), dtype=torch.uint8).to(device)
        image = ImageData.from_tensor(pixels)

        restored = IMAGE_KIND.to_payload(IMAGE_KIND.to_serialized(image))

        assert image.device == pixels.device
        assert restored.device == torch.device("cpu")
        assert torch.equal(restored.tensor_image, pixels.to("cpu"))

    def test_bare_encoded_image_is_rgb_with_a_new_identity(self) -> None:
        bgr = _random_bgr(6, 7, seed=3)
        encoded = base64.b64encode(cv2.imencode(".png", bgr)[1].tobytes()).decode()
        value = {"type": "base64", "value": encoded}

        first, second = IMAGE_KIND.to_payload(value), IMAGE_KIND.to_payload(value)

        assert torch.equal(
            first.tensor_image,
            torch.from_numpy(bgr[:, :, ::-1].transpose(2, 0, 1).copy()),
        )
        assert first.image_id != second.image_id

    def test_wire_with_image_id_but_no_mappings_is_rejected(self) -> None:
        image = ImageData.from_tensor(torch.zeros((3, 2, 2), dtype=torch.uint8))
        wire = IMAGE_KIND.to_serialized(image)
        del wire["root"]

        with pytest.raises(ContractError, match="root"):
            IMAGE_KIND.to_payload(wire)


class TestV1Ingress:
    @staticmethod
    def _v1_chain():
        bgr = _random_bgr(80, 120, seed=5)
        root = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="camera"),
            numpy_image=bgr,
            video_metadata=_video_metadata(),
        )
        crop = WorkflowImageData.create_crop(
            root,
            crop_identifier="det-1",
            cropped_image=bgr[20:60, 30:90].copy(),
            offset_x=30,
            offset_y=20,
        )
        nested = WorkflowImageData.create_crop(
            crop,
            crop_identifier="det-2",
            cropped_image=bgr[27:47, 35:75].copy(),
            offset_x=5,
            offset_y=7,
        )

        return bgr, root, crop, nested

    def test_root_is_converted_from_bgr_and_keeps_its_id_and_video(self) -> None:
        bgr, root, _, _ = self._v1_chain()

        image = IMAGE_KIND.to_payload(root)

        expected = torch.from_numpy(bgr[:, :, ::-1].transpose(2, 0, 1).copy())
        assert torch.equal(image.tensor_image, expected)
        assert image.image_id == image.parent.frame_id == image.root.frame_id
        assert image.image_id == "camera"
        assert image.video_metadata is root._video_metadata

    def test_crops_match_the_v1_metadata_oracle(self) -> None:
        _, root, crop, nested = self._v1_chain()

        for v1_image in (root, crop, nested):
            oracle = build_native_image_metadata(
                v1_image, class_names={}, prediction_type="object-detection"
            )
            metadata = ImageData.from_workflow_image_data(
                v1_image
            ).prediction_metadata()

            assert {key: metadata[key] for key in V1_KEYS} == {
                key: oracle[key] for key in V1_KEYS
            }
            assert metadata["scaling_relative_to_root_parent"] == 1.0

    def test_v1_crops_do_not_invent_an_ancestor_id(self) -> None:
        _, _, crop, nested = self._v1_chain()

        adapted = [
            ImageData.from_workflow_image_data(image) for image in (crop, nested)
        ]

        assert [image.image_id for image in adapted] == ["det-1", "det-2"]
        assert [image.parent.frame_id for image in adapted] == [None, None]
        assert adapted[1].root == FrameMapping("camera", (80, 120), offset_xy=(35, 27))
        assert adapted[1].prediction_metadata()["parent_frame_id"] is None

    def test_absent_video_metadata_stays_absent(self) -> None:
        v1_image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="still"),
            numpy_image=np.zeros((2, 2, 3), dtype=np.uint8),
        )

        assert ImageData.from_workflow_image_data(v1_image).video_metadata is None

    @pytest.mark.parametrize("device", DEVICES)
    def test_materialized_v1_tensor_is_used_as_is(self, device) -> None:
        pixels = torch.zeros((3, 10, 12), dtype=torch.uint8, device=device)
        root = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="tensor-root"),
            tensor_image=pixels,
        )
        crop = WorkflowImageData.create_crop_from_tensor(
            root,
            crop_identifier="det",
            cropped_tensor_image=pixels[:, 2:6, 3:9],
            offset_x=3,
            offset_y=2,
        )

        adapted = ImageData.from_workflow_image_data(crop)

        assert adapted.tensor_image is crop.tensor_image
        assert adapted.device == pixels.device
        assert adapted.root.offset_xy == (3.0, 2.0)

    def test_v1_grayscale_numpy_becomes_one_channel(self) -> None:
        v1_image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="gray"),
            numpy_image=np.full((3, 4), 5, dtype=np.uint8),
        )

        adapted = ImageData.from_workflow_image_data(v1_image)

        assert (adapted.channels, adapted.size_hw) == (1, (3, 4))

    def test_v1_serialized_image_reads_like_the_v1_container(self) -> None:
        _, _, _, nested = self._v1_chain()

        wire = serialise_image(nested)
        restored = IMAGE_KIND.to_payload(wire)
        adapted = ImageData.from_workflow_image_data(nested)

        # V1 writes JPEG, so compare with V1's own bytes decoded as RGB.
        jpeg = np.frombuffer(base64.b64decode(wire["value"]), dtype=np.uint8)
        rgb = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)[:, :, ::-1]
        assert torch.equal(
            restored.tensor_image, torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        )
        assert (restored.image_id, restored.parent, restored.root) == (
            adapted.image_id,
            adapted.parent,
            adapted.root,
        )
        # V1 crops drop video metadata and V1's serializer invents one.
        assert restored.video_metadata == VideoMetadata.model_validate(
            wire["video_metadata"]
        )


class TestEngineFlow:
    @pytest.mark.parametrize("device", DEVICES)
    def test_compiled_workflow_keeps_tensors_on_device_without_host_copies(
        self, device, host_transfers_forbidden
    ) -> None:
        definition = {
            "version": "2.0",
            "inputs": [
                {"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}
            ],
            "steps": [
                {
                    "type": "v2/crop",
                    "name": "crop",
                    "image": "$inputs.images",
                    "regions": [[2, 3, 30, 20], [10, 0, 40, 40]],
                },
                {
                    "type": "v2/resize",
                    "name": "resize",
                    "image": "$steps.crop.crops",
                    "width": 9,
                    "height": 5,
                },
                {"type": "v2/invert", "name": "invert", "image": "$steps.resize.image"},
                {
                    "type": "v2/mosaic",
                    "name": "mosaic",
                    "images": "$steps.invert.image",
                    "tile_size": 4,
                },
            ],
            "outputs": [
                {"type": "JsonField", "name": "images", "selector": "$inputs.images"},
                {"type": "JsonField", "name": "crops", "selector": "$steps.crop.crops"},
                {
                    "type": "JsonField",
                    "name": "inverted",
                    "selector": "$steps.invert.image",
                },
                {
                    "type": "JsonField",
                    "name": "mosaic",
                    "selector": "$steps.mosaic.image",
                },
            ],
        }
        pixels = [
            torch.zeros((3, 30, 50), dtype=torch.uint8, device=device),
            torch.zeros((1, 25, 35), dtype=torch.uint8, device=device),
        ]
        plan = compile_workflow(definition, catalogue=create_catalogue())

        rows = plan.create_session().run({"images": pixels}).rows()

        for row, given in zip(rows, pixels):
            assert row["images"].tensor_image is given
        images = [row["mosaic"] for row in rows]
        for row in rows:
            images.extend(row["crops"] + row["inverted"])
        assert len(images) == 2 + 4 + 4
        assert {image.device for image in images} == {pixels[0].device}
