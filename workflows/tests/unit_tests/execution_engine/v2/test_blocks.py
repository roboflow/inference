"""Tests of the native V2 image blocks, their media kinds and their catalogue.

Blocks are exercised the way an author tests them: step parameters are
validated by the class's own declaration, exactly as the compiler does, and
``run`` is then called with the validated values. Groups are passed as the
``Batch`` objects the engine builds, including sparse and empty ones.
"""

import base64
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
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
BLOCK_CLASSES = (CropBlock, InvertBlock, MosaicBlock, HasBrightnessBlock)


def _image(height: int, width: int, *, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    image = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    return image


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
    def test_accepts_rgb_uint8(self) -> None:
        IMAGE_KIND.check(_image(4, 5))

    @pytest.mark.parametrize(
        "payload",
        [
            np.zeros((4, 5, 3), dtype=np.float32),
            np.zeros((4, 5), dtype=np.uint8),
            np.zeros((4, 5, 4), dtype=np.uint8),
            np.zeros((0, 5, 3), dtype=np.uint8),
            [[0, 0, 0]],
            None,
        ],
    )
    def test_rejects_other_payloads(self, payload) -> None:
        with pytest.raises(ContractError, match="not a valid 'image'"):
            IMAGE_KIND.check(payload)

    def test_serialization_round_trip_is_pixel_exact(self) -> None:
        image = _image(7, 9)

        serialized = IMAGE_KIND.to_serialized(image)
        restored = IMAGE_KIND.to_payload(serialized)

        assert serialized["type"] == "base64"
        assert isinstance(serialized["value"], str)
        np.testing.assert_array_equal(restored, image)
        IMAGE_KIND.check(restored)

    def test_array_input_passes_through_unchanged(self) -> None:
        image = _image(3, 3)

        assert IMAGE_KIND.to_payload(image) is image

    @pytest.mark.parametrize(
        "value, fragment",
        [
            ("image.jpg", "must be an RGB uint8 numpy array"),
            ({"type": "url", "value": "x"}, "must be an RGB uint8 numpy array"),
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

    def test_clipping_and_pixel_equality(self) -> None:
        image = _image(50, 60)

        result = _run(
            CropBlock,
            data={"image": image},
            regions=[[-10, -10, 20, 30], [50, 40, 100, 100]],
        )

        assert len(result["crops"]) == 2
        np.testing.assert_array_equal(result["crops"][0], image[0:30, 0:20])
        np.testing.assert_array_equal(result["crops"][1], image[40:50, 50:60])
        assert result["summary"]["crop_dimensions"] == [[30, 20], [10, 10]]

    def test_zero_area_rectangle_and_no_rectangles_give_empty_batches(self) -> None:
        for regions in ([[5, 5, 5, 20]], []):
            result = _run(CropBlock, data={"image": _image(30, 30)}, regions=regions)

            assert isinstance(result["crops"], Batch)
            assert len(result["crops"]) == 0
            assert result["summary"]["crop_count"] == 0

    def test_crops_are_independent_contiguous_copies(self) -> None:
        image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        original = image.copy()
        regions = [[0, 0, 5, 3], [20, 20, 21, 21], [0, 1, 5, 3], [0, 0, 3, 2]]

        crops = _run(CropBlock, data={"image": image}, regions=regions)["crops"]

        assert crops.indices == ((0,), (2,), (3,))
        expected = [original, original[1:], original[:2, :3]]
        for position, crop in enumerate(crops):
            np.testing.assert_array_equal(crop, expected[position])
            assert crop.flags.c_contiguous
            assert not np.shares_memory(crop, image)
            for other in crops.content[position + 1 :]:
                assert not np.shares_memory(crop, other)

        crops[0][...] = 99
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(crops[1], original[1:])


class TestInvertBlock:
    def test_pixel_inversion_leaves_input_untouched(self) -> None:
        image = _image(8, 9)
        original = image.copy()

        result = _run(InvertBlock, data={"image": image})

        np.testing.assert_array_equal(result["image"], 255 - original)
        np.testing.assert_array_equal(image, original)
        IMAGE_KIND.check(result["image"])


class TestMosaicBlock:
    def test_empty_group_yields_blank_canvas_and_zero(self) -> None:
        empty_group = Batch.empty(parent_index=(1,))

        result = _run(
            MosaicBlock, data={"images": empty_group}, tile_size=16, background=7
        )

        assert result["count"] == 0
        assert result["image"].shape == (16, 16, 3)
        assert int(result["image"].min()) == 7 and int(result["image"].max()) == 7

    def test_two_constant_images_fill_two_tiles(self) -> None:
        red = np.full((10, 20, 3), (200, 0, 0), dtype=np.uint8)
        blue = np.full((30, 5, 3), (0, 0, 200), dtype=np.uint8)

        result = _run(MosaicBlock, data={"images": Batch.of([red, blue])}, tile_size=8)

        assert result["count"] == 2
        assert result["image"].shape == (8, 16, 3)
        np.testing.assert_array_equal(
            result["image"][:, :8], np.broadcast_to(red[0, 0], (8, 8, 3))
        )
        np.testing.assert_array_equal(
            result["image"][:, 8:], np.broadcast_to(blue[0, 0], (8, 8, 3))
        )

    def test_sparse_group_is_tiled_in_order_without_gaps(self) -> None:
        values = (10, 30)
        survivors = Batch(
            [np.full((4, 4, 3), value, dtype=np.uint8) for value in values],
            indices=((2, 0), (2, 2)),
            parent_index=(2,),
        )

        result = _run(
            MosaicBlock, data={"images": survivors}, tile_size=4, background=255
        )

        assert result["count"] == 2
        assert result["image"].shape == (4, 8, 3)
        assert [int(result["image"][0, column, 0]) for column in (0, 4)] == [10, 30]

    def test_columns_parameter_controls_grid(self) -> None:
        images = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (10, 20, 30)]

        result = _run(
            MosaicBlock, data={"images": Batch.of(images)}, tile_size=4, columns=1
        )

        assert result["image"].shape == (12, 4, 3)
        assert [int(result["image"][row * 4, 0, 0]) for row in range(3)] == [10, 20, 30]

    def test_partial_last_row_keeps_background(self) -> None:
        images = [np.full((2, 2, 3), 50, dtype=np.uint8)] * 3

        result = _run(
            MosaicBlock, data={"images": Batch.of(images)}, tile_size=2, background=9
        )

        assert result["image"].shape == (4, 4, 3)
        assert int(result["image"][3, 3, 0]) == 9


class TestHasBrightnessBlock:
    def test_threshold_is_inclusive_and_returns_bool(self) -> None:
        exact = np.full((3, 3, 3), 100, dtype=np.uint8)
        dark = np.full((3, 3, 3), 99, dtype=np.uint8)

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
