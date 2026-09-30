"""Tests of the native V2 image blocks, their media kinds and their catalogue.

Blocks are exercised the way an author tests them: step parameters are
validated by the class's own declaration, exactly as the compiler does, and
``run`` is then called with the validated values. Groups are passed as the
``Batch`` objects the engine builds, including sparse and empty ones. Image
payloads are ``ImageData``; the carrier itself is covered in
``test_tensor_image_data.py``.
"""

import base64
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks import (
    CROP_SUMMARY_KIND,
    IMAGE_KIND,
    ContinueIfBlock,
    CropBlock,
    HasBrightnessBlock,
    InvertBlock,
    MosaicBlock,
    create_catalogue,
)
from roboflow_workflows.execution_engine.v2.blocks.image import ResizeBlock
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import Select, Stop, spec_of
from roboflow_workflows.execution_engine.v2.errors import (
    CatalogueError,
    ContractError,
    ParamsValidationError,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    Kind,
)

REGIONS = [[40, 40, 100, 100], [120, 10, 180, 70]]
BLOCK_CLASSES = (CropBlock, ResizeBlock, InvertBlock, MosaicBlock, HasBrightnessBlock)


def _image(height: int, width: int, *, seed: int = 0) -> ImageData:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(
        0, 256, (3, height, width), dtype=torch.uint8, generator=generator
    )

    return ImageData.from_tensor(pixels)


def _uniform(height: int, width: int, value, *, channels: int = 3) -> ImageData:
    color = torch.tensor(value, dtype=torch.uint8).reshape(-1, 1, 1)
    pixels = color.expand(channels, height, width).contiguous()

    return ImageData.from_tensor(pixels)


def _three_images() -> list:
    images = [
        _image(120, 200, seed=1),
        _image(32, 32, seed=2),
        _image(110, 110, seed=3),
    ]

    return images


def _run(block_class: type, *, data: Dict[str, Any], **parameters: Any) -> Any:
    """Run a fresh block the way one step invocation does.

    ``parameters`` are validated by the block's declaration. Each ``data`` field
    is declared as a selector and replaced by its value, as the engine does.
    """
    selectors = {name: f"$inputs.{name}" for name in data}
    params = spec_of(block_class).validate_params({**parameters, **selectors})
    arguments = {**dict(params), **data}

    result = block_class().run(**arguments)

    return result


class TestImageKind:
    def test_accepts_image_data_only(self) -> None:
        IMAGE_KIND.check(_image(4, 5))

    @pytest.mark.parametrize(
        "payload",
        [
            torch.zeros((3, 4, 5), dtype=torch.uint8),
            np.zeros((4, 5, 3), dtype=np.uint8),
            [[0, 0, 0]],
            None,
        ],
    )
    def test_rejects_raw_pixels_and_other_payloads(self, payload) -> None:
        with pytest.raises(ContractError, match="not a valid 'image'"):
            IMAGE_KIND.check(payload)

    def test_serialization_round_trip_is_pixel_exact(self) -> None:
        image = _image(7, 9)

        serialized = IMAGE_KIND.to_serialized(image)
        restored = IMAGE_KIND.to_payload(serialized)

        assert serialized["type"] == "base64"
        assert isinstance(serialized["value"], str)
        assert torch.equal(restored.tensor_image, image.tensor_image)
        assert restored.image_id == image.image_id
        IMAGE_KIND.check(restored)

    def test_image_input_passes_through_unchanged(self) -> None:
        image = _image(3, 3)

        assert IMAGE_KIND.to_payload(image) is image

    def test_prior_rgb_array_input_is_converted_once(self) -> None:
        array = np.zeros((2, 3, 3), dtype=np.uint8)
        array[..., 0] = 200

        image = IMAGE_KIND.to_payload(array)

        assert image.size_hw == (2, 3)
        assert image.tensor_image[:, 0, 0].tolist() == [200, 0, 0]

    @pytest.mark.parametrize(
        "value, fragment",
        [
            ("image.jpg", "must be ImageData, a"),
            ({"type": "url", "value": "x"}, "must be ImageData, a"),
            ({"type": "base64", "value": "%%%"}, "invalid base64"),
            (
                {"type": "base64", "value": base64.b64encode(b"text").decode()},
                "does not decode to an image",
            ),
        ],
    )
    def test_rejects_undecodable_input(self, value, fragment) -> None:
        with pytest.raises(ContractError, match=fragment):
            IMAGE_KIND.to_payload(value)


class TestCropSummaryKind:
    def test_requires_integer_counts_and_dimensions(self) -> None:
        CROP_SUMMARY_KIND.check({"crop_count": 1, "image_height": 2, "image_width": 3})

        for payload in (
            {"crop_count": 1},
            {"crop_count": True, "image_height": 2, "image_width": 3},
            [1, 2, 3],
        ):
            with pytest.raises(ContractError):
                CROP_SUMMARY_KIND.check(payload)


class TestDeclarations:
    def test_constructors_declare_no_resources(self) -> None:
        for block_class in BLOCK_CLASSES + (ContinueIfBlock,):
            assert spec_of(block_class).resources == ()

    def test_crop_expands_crops_and_keeps_summary_per_image(self) -> None:
        spec = spec_of(CropBlock)
        crops, summary = spec.outputs["crops"], spec.outputs["summary"]

        assert spec.type == "v2/crop"
        assert (crops.transform, crops.expand, crops.stationary) == (
            "expand",
            "regions",
            False,
        )
        assert crops.kinds == (IMAGE_KIND,)
        assert summary.transform == "same"
        assert summary.kinds == (CROP_SUMMARY_KIND,)
        assert {crops.source, summary.source} == {"image"}

    def test_resize_outputs_one_image_per_image(self) -> None:
        spec = spec_of(ResizeBlock)

        assert spec.type == "v2/resize"
        assert spec.outputs["image"].transform == "same"
        assert spec.outputs["image"].kinds == (IMAGE_KIND,)
        assert spec.fields["width"].whole.kinds == (INTEGER_KIND,)
        params = spec.validate_params(
            {"image": "$inputs.i", "width": 4, "height": "$inputs.h"}
        )
        assert params.interpolation == "bilinear"

    def test_mosaic_consumes_a_group_and_outputs_one_value_per_parent(self) -> None:
        spec = spec_of(MosaicBlock)

        assert spec.fields["images"].role == "group"
        assert spec.fields["images"].whole.kinds == (IMAGE_KIND,)
        for output in spec.outputs.values():
            assert output.transform == "same"
            assert (output.source, output.context_policy) == (
                "images",
                "common_or_none",
            )
        assert spec.outputs["count"].kinds == (INTEGER_KIND,)

    def test_plain_values_use_builtin_kind_objects(self) -> None:
        assert spec_of(HasBrightnessBlock).outputs["keep"].kinds == (BOOLEAN_KIND,)
        assert spec_of(HasBrightnessBlock).fields["minimum"].whole.kinds == (
            FLOAT_KIND,
        )
        assert spec_of(MosaicBlock).fields["tile_size"].whole.kinds == (INTEGER_KIND,)
        assert spec_of(CropBlock).fields["regions"].whole.kinds == (
            LIST_OF_VALUES_KIND,
        )
        assert spec_of(ContinueIfBlock).fields["condition"].whole.kinds == (
            BOOLEAN_KIND,
        )

    def test_continue_if_is_an_output_free_control_block(self) -> None:
        spec = spec_of(ContinueIfBlock)

        assert spec.is_control
        assert dict(spec.outputs) == {}
        assert spec.fields["next_steps"].role == "step"

    def test_every_run_parameter_accepts_selectors(self) -> None:
        spec = spec_of(MosaicBlock)
        params = spec.validate_params(
            {
                "images": "$steps.crop.crops",
                "tile_size": "$inputs.tile_size",
                "columns": "$steps.layout.columns",
                "background": 12,
            }
        )

        uses = {use.field: use.selector for use in spec.find_selectors(params)}

        assert uses == {
            "images": "$steps.crop.crops",
            "tile_size": "$inputs.tile_size",
            "columns": "$steps.layout.columns",
        }
        assert params.background == 12

    def test_regions_accept_a_selector_or_a_literal(self) -> None:
        spec = spec_of(CropBlock)

        bound = spec.validate_params({"image": "$inputs.image", "regions": "$inputs.r"})
        literal = spec.validate_params({"image": "$inputs.image", "regions": REGIONS})

        assert [use.field for use in spec.find_selectors(bound)] == ["image", "regions"]
        assert [use.field for use in spec.find_selectors(literal)] == ["image"]
        assert literal.regions == [(40, 40, 100, 100), (120, 10, 180, 70)]

    def test_mosaic_defaults_are_not_explicit_values(self) -> None:
        params = spec_of(MosaicBlock).validate_params({"images": "$steps.c.crops"})

        assert (params.tile_size, params.columns, params.background) == (64, None, 0)
        assert params.model_fields_set == {"images"}

    @pytest.mark.parametrize(
        "block_class, parameters, fragment",
        [
            (CropBlock, {"image": "$inputs.i"}, "regions: Field required"),
            (CropBlock, {"image": "$inputs.i", "regions": "no"}, "valid list"),
            (CropBlock, {"image": "$inputs.i", "regions": [[0, 0, 10]]}, "required"),
            (
                CropBlock,
                {"image": "$inputs.i", "regions": [[0, 0, 10, "a"]]},
                "valid integer",
            ),
            (
                CropBlock,
                {"image": "$inputs.i", "regions": [[0, 0, 10, True]]},
                "valid integer",
            ),
            (
                CropBlock,
                {"image": "$inputs.i", "regions": [[10, 0, 0, 10]]},
                "x0 <= x1",
            ),
            (
                CropBlock,
                {"image": "$inputs.i", "regions": [], "padding": 2},
                "padding: Extra inputs are not permitted",
            ),
            (ResizeBlock, {"image": "$inputs.i", "width": 4}, "height: Field required"),
            (
                ResizeBlock,
                {"image": "$inputs.i", "width": 0, "height": 4},
                "greater than",
            ),
            (
                ResizeBlock,
                {
                    "image": "$inputs.i",
                    "width": 4,
                    "height": 4,
                    "interpolation": "cubic",
                },
                "interpolation",
            ),
            (InvertBlock, {"image": "$inputs.i", "strength": 2}, "strength"),
            (MosaicBlock, {"images": "$inputs.i", "tile_size": 0}, "greater than"),
            (MosaicBlock, {"images": "$inputs.i", "tile_size": True}, "integer"),
            (MosaicBlock, {"images": "$inputs.i", "columns": -1}, "greater than"),
            (MosaicBlock, {"images": "$inputs.i", "background": 256}, "less than"),
            (MosaicBlock, {"images": "$inputs.i", "size": 3}, "size"),
            (HasBrightnessBlock, {"image": "$inputs.i"}, "minimum: Field required"),
            (HasBrightnessBlock, {"image": "$inputs.i", "minimum": 300}, "less than"),
            (HasBrightnessBlock, {"image": "$inputs.i", "minimum": "5"}, "number"),
            (HasBrightnessBlock, {"image": "$inputs.i", "minimum": True}, "number"),
            (ContinueIfBlock, {"condition": 1, "next_steps": []}, "boolean"),
        ],
    )
    def test_invalid_parameters_name_the_block_and_field(
        self, block_class, parameters, fragment
    ) -> None:
        with pytest.raises(ParamsValidationError) as info:
            spec_of(block_class).validate_params(parameters, step_path=("my_step",))

        message = str(info.value)
        assert fragment in message
        assert spec_of(block_class).type in message
        assert "my_step" in message

    @pytest.mark.parametrize(
        "block_class, field, value",
        [
            (CropBlock, "regions", [[5, 0, 0, 5]]),
            (CropBlock, "regions", [[0, 0, 5]]),
            (ResizeBlock, "width", 0),
            (MosaicBlock, "tile_size", 0),
            (MosaicBlock, "background", -1),
            (HasBrightnessBlock, "minimum", 256.0),
        ],
    )
    def test_selected_values_obey_the_literal_constraints(
        self, block_class, field, value
    ) -> None:
        with pytest.raises(ContractError, match=field):
            spec_of(block_class).validate_resolved_value(field, value)

    def test_selected_regions_are_checked_but_not_converted(self) -> None:
        selected = [[0, 0, 5, 5]]

        validated = spec_of(CropBlock).validate_resolved_value("regions", selected)
        literal = spec_of(CropBlock).validate_params(
            {"image": "$inputs.i", "regions": [[0, 0, 5, 5]]}
        )

        assert validated is selected
        assert literal.regions == [(0, 0, 5, 5)]


class TestCropBlock:
    def test_ragged_counts_and_sparse_local_indices(self) -> None:
        results = [
            _run(CropBlock, data={"image": image}, regions=REGIONS)
            for image in _three_images()
        ]

        assert [len(result["crops"]) for result in results] == [2, 0, 1]
        assert results[0]["crops"].indices == ((0,), (1,))
        assert results[1]["crops"].indices == ()
        assert results[2]["crops"].indices == ((0,),)
        assert [result["summary"]["crop_count"] for result in results] == [2, 0, 1]
        for result in results:
            CROP_SUMMARY_KIND.check(result["summary"])

    def test_omitted_second_region_keeps_configured_index_of_third(self) -> None:
        regions = [[0, 0, 10, 10], [500, 500, 600, 600], [10, 10, 20, 20]]

        result = _run(CropBlock, data={"image": _image(40, 40)}, regions=regions)

        assert result["crops"].indices == ((0,), (2,))
        assert result["summary"]["kept_regions"] == [0, 2]

    def test_clipping_pixel_equality_and_provenance(self) -> None:
        image = _image(50, 60)

        result = _run(
            CropBlock,
            data={"image": image},
            regions=[[-10, -10, 20, 30], [50, 40, 100, 100]],
        )

        first, second = result["crops"]
        assert torch.equal(first.tensor_image, image.tensor_image[:, 0:30, 0:20])
        assert torch.equal(second.tensor_image, image.tensor_image[:, 40:50, 50:60])
        assert result["summary"]["crop_dimensions"] == [[30, 20], [10, 10]]
        assert first.parent.offset_xy == (0.0, 0.0)
        assert second.parent.offset_xy == (50.0, 40.0)
        for crop in (first, second):
            assert crop.parent.frame_id == image.image_id
            assert crop.root.frame_id == image.image_id
            assert crop.image_id not in (image.image_id, "")
        assert first.image_id != second.image_id

    def test_zero_area_rectangle_and_no_rectangles_give_empty_batches(self) -> None:
        for regions in ([[5, 5, 5, 20]], []):
            result = _run(CropBlock, data={"image": _image(30, 30)}, regions=regions)

            assert isinstance(result["crops"], Batch)
            assert len(result["crops"]) == 0
            assert result["summary"]["crop_count"] == 0

    def test_crops_are_independent_contiguous_copies(self) -> None:
        pixels = torch.arange(3 * 3 * 5, dtype=torch.uint8).reshape(3, 3, 5)
        image = ImageData.from_tensor(pixels)
        original = pixels.clone()
        regions = [[0, 0, 5, 3], [20, 20, 21, 21], [0, 1, 5, 3], [0, 0, 3, 2]]

        crops = _run(CropBlock, data={"image": image}, regions=regions)["crops"]

        assert crops.indices == ((0,), (2,), (3,))
        expected = [original, original[:, 1:], original[:, :2, :3]]
        storages = {pixels.untyped_storage().data_ptr()}
        for position, crop in enumerate(crops):
            assert torch.equal(crop.tensor_image, expected[position])
            assert crop.tensor_image.is_contiguous()
            storages.add(crop.tensor_image.untyped_storage().data_ptr())
        assert len(storages) == 4

        crops[0].tensor_image[...] = 99
        assert torch.equal(pixels, original)
        assert torch.equal(crops[1].tensor_image, original[:, 1:])


class TestResizeBlock:
    def test_resize_keeps_device_and_records_the_actual_ratio(self) -> None:
        image = _image(10, 7)

        resized = _run(ResizeBlock, data={"image": image}, width=3, height=4)["image"]

        assert resized.size_hw == (4, 3)
        assert resized.device == image.device
        assert resized.parent.frame_id == image.image_id
        assert resized.parent.scale_xy == (7 / 3, 10 / 4)
        assert resized.root.scale_xy == (7 / 3, 10 / 4)

    def test_nearest_keeps_the_former_sampling_rule(self) -> None:
        image = _image(9, 13, seed=4)

        resized = _run(
            ResizeBlock,
            data={"image": image},
            width=5,
            height=4,
            interpolation="nearest",
        )["image"]

        rows = (np.arange(4) * 9) // 4
        columns = (np.arange(5) * 13) // 5
        expected = image.tensor_image.numpy()[:, rows[:, None], columns[None, :]]
        assert np.array_equal(resized.tensor_image.numpy(), expected)


class TestInvertBlock:
    def test_pixel_inversion_keeps_provenance_and_leaves_input_untouched(self) -> None:
        image = _image(8, 9)
        original = image.tensor_image.clone()

        result = _run(InvertBlock, data={"image": image})

        inverted = result["image"]
        assert torch.equal(inverted.tensor_image, 255 - original)
        assert torch.equal(image.tensor_image, original)
        assert (inverted.image_id, inverted.parent, inverted.root) == (
            image.image_id,
            image.parent,
            image.root,
        )
        IMAGE_KIND.check(inverted)


class TestMosaicBlock:
    def test_empty_group_yields_blank_composite_canvas_and_zero(self) -> None:
        empty_group = Batch.empty(parent_index=(1,))

        result = _run(
            MosaicBlock, data={"images": empty_group}, tile_size=16, background=7
        )

        canvas = result["image"]
        assert result["count"] == 0
        assert tuple(canvas.tensor_image.shape) == (3, 16, 16)
        assert int(canvas.tensor_image.min()) == int(canvas.tensor_image.max()) == 7
        assert canvas.is_composite
        assert canvas.composite_sources == ()
        assert canvas.prediction_metadata()["composite_sources"] == []

    def test_two_constant_images_fill_two_tiles(self) -> None:
        red = _uniform(10, 20, (200, 0, 0))
        blue = _uniform(30, 5, (0, 0, 200))

        result = _run(MosaicBlock, data={"images": Batch.of([red, blue])}, tile_size=8)

        pixels = result["image"].tensor_image
        assert result["count"] == 2
        assert tuple(pixels.shape) == (3, 8, 16)
        assert torch.equal(pixels[:, :, :8], red.tensor_image[:, :8, :8])
        assert torch.equal(
            pixels[:, :, 8:], blue.tensor_image[:, :8, :1].expand(3, 8, 8)
        )

    def test_canvas_is_a_new_root_with_per_tile_provenance(self) -> None:
        red = _uniform(10, 20, (200, 0, 0))
        blue = _uniform(30, 5, (0, 0, 200))

        canvas = _run(MosaicBlock, data={"images": Batch.of([red, blue])}, tile_size=8)[
            "image"
        ]

        assert canvas.parent.frame_id == canvas.root.frame_id == canvas.image_id
        assert canvas.image_id not in (red.image_id, blue.image_id)
        first, second = canvas.composite_sources
        assert (first.index, first.image_id, first.canvas_xyxy) == (
            (0,),
            red.image_id,
            (0, 0, 8, 8),
        )
        assert (second.index, second.image_id, second.canvas_xyxy) == (
            (1,),
            blue.image_id,
            (8, 0, 16, 8),
        )
        assert second.parent.map_xy(8, 0) == (0.0, 0.0)
        assert second.parent.map_xy(16, 8) == (5.0, 30.0)

    def test_sparse_group_is_tiled_in_order_without_gaps(self) -> None:
        values = (10, 30)
        survivors = Batch(
            [_uniform(4, 4, value) for value in values],
            indices=((2, 0), (2, 2)),
            parent_index=(2,),
        )

        result = _run(
            MosaicBlock, data={"images": survivors}, tile_size=4, background=255
        )

        pixels = result["image"].tensor_image
        assert result["count"] == 2
        assert tuple(pixels.shape) == (3, 4, 8)
        assert [int(pixels[0, 0, column]) for column in (0, 4)] == [10, 30]
        assert [source.index for source in result["image"].composite_sources] == [
            (2, 0),
            (2, 2),
        ]

    def test_columns_parameter_controls_grid(self) -> None:
        images = [_uniform(4, 4, value) for value in (10, 20, 30)]

        result = _run(
            MosaicBlock, data={"images": Batch.of(images)}, tile_size=4, columns=1
        )

        pixels = result["image"].tensor_image
        assert tuple(pixels.shape) == (3, 12, 4)
        assert [int(pixels[0, row * 4, 0]) for row in range(3)] == [10, 20, 30]

    def test_partial_last_row_keeps_background(self) -> None:
        images = [_uniform(2, 2, 50)] * 3

        result = _run(
            MosaicBlock, data={"images": Batch.of(images)}, tile_size=2, background=9
        )

        pixels = result["image"].tensor_image
        assert tuple(pixels.shape) == (3, 4, 4)
        assert int(pixels[0, 3, 3]) == 9

    def test_grayscale_tiles_repeat_on_an_rgb_canvas(self) -> None:
        gray = _uniform(3, 3, 80, channels=1)
        color = _uniform(3, 3, (1, 2, 3))

        mixed = _run(
            MosaicBlock, data={"images": Batch.of([gray, color])}, tile_size=2
        )["image"]
        only_gray = _run(MosaicBlock, data={"images": Batch.of([gray])}, tile_size=2)[
            "image"
        ]

        assert mixed.channels == 3
        assert mixed.tensor_image[:, 0, 0].tolist() == [80, 80, 80]
        assert mixed.tensor_image[:, 0, 2].tolist() == [1, 2, 3]
        assert only_gray.channels == 1

    def test_images_on_different_devices_are_rejected(self) -> None:
        on_cpu = _uniform(2, 2, 1)
        elsewhere = ImageData.from_tensor(
            torch.empty((3, 2, 2), dtype=torch.uint8, device="meta")
        )

        with pytest.raises(ValueError, match="one device.*cpu.*meta"):
            _run(MosaicBlock, data={"images": Batch.of([on_cpu, elsewhere])})


class TestHasBrightnessBlock:
    def test_threshold_is_inclusive_and_returns_bool(self) -> None:
        exact = _uniform(3, 3, 100)
        dark = _uniform(3, 3, 99)

        assert (
            _run(HasBrightnessBlock, data={"image": exact}, minimum=100)["keep"] is True
        )
        assert (
            _run(HasBrightnessBlock, data={"image": dark}, minimum=100)["keep"] is False
        )


class TestContinueIfBlock:
    def test_true_selects_every_next_step_and_false_stops(self) -> None:
        next_steps = ["$steps.invert", "$steps.mosaic"]

        admitted = _run(ContinueIfBlock, data={}, condition=True, next_steps=next_steps)
        denied = _run(ContinueIfBlock, data={}, condition=False, next_steps=next_steps)

        assert admitted == Select(next_steps)
        assert denied == Stop()


class TestCatalogue:
    def test_collects_the_block_classes_and_media_kinds(self) -> None:
        catalogue = create_catalogue()

        assert catalogue.block_types == (
            "v2/crop",
            "v2/resize",
            "v2/invert",
            "v2/mosaic",
            "v2/has_brightness",
            "v2/continue_if",
        )
        assert catalogue.entry("v2/crop").spec is spec_of(CropBlock)
        assert catalogue.kind("image") is IMAGE_KIND
        assert catalogue.kind("crop_summary") is CROP_SUMMARY_KIND
        assert catalogue.kind("boolean") is BOOLEAN_KIND

    def test_plugin_convention_reads_the_same_catalogue(self) -> None:
        loaded = Catalogue.from_modules(
            ["roboflow_workflows.execution_engine.v2.blocks"]
        )

        assert loaded.block_types == create_catalogue().block_types

    def test_merges_with_blocks_that_share_builtin_kinds(self) -> None:
        other = Catalogue(kinds=[BOOLEAN_KIND, INTEGER_KIND])

        merged = Catalogue.merge(create_catalogue(), other)

        assert merged.kind("boolean") is BOOLEAN_KIND

    def test_rejects_a_second_kind_with_a_builtin_name(self) -> None:
        private_boolean = Catalogue(kinds=[Kind(name="boolean")])

        with pytest.raises(CatalogueError, match="boolean"):
            Catalogue.merge(create_catalogue(), private_boolean)


class TestIsolation:
    def test_generic_core_imports_no_image_dependency(self) -> None:
        script = (
            "import sys\n"
            "import roboflow_workflows.execution_engine.v2.catalogue\n"
            "import roboflow_workflows.execution_engine.v2.data\n"
            "import roboflow_workflows.execution_engine.v2.declaration\n"
            "import roboflow_workflows.execution_engine.v2.kinds\n"
            "import roboflow_workflows.execution_engine.v2.plan\n"
            "loaded = [m for m in sys.modules if m == 'cv2' or m.startswith("
            "'roboflow_workflows.execution_engine.v2.blocks')]\n"
            "print(loaded)\n"
            "sys.exit(1 if loaded else 0)\n"
        )
        workflows_root = Path(__file__).resolve().parents[4]

        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True,
            text=True,
            cwd=str(workflows_root),
            env={"PYTHONPATH": str(workflows_root), "PYTHONDONTWRITEBYTECODE": "1"},
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_legacy_loader_does_not_discover_v2_blocks(self) -> None:
        loader_path = (
            Path(__file__).resolve().parents[4]
            / "roboflow_workflows"
            / "core_steps"
            / "loader.py"
        )

        assert "execution_engine.v2" not in loader_path.read_text()
