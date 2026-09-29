"""Tests of the native V2 CPU image blocks and their explicit registry.

The first part exercises the blocks directly, the way an author would in a
unit test. The last part compiles small definitions through the real V2
engine to check that the blocks behave inside the graph: ragged crop counts
``[2, 0, 1]``, per-output layouts, the blank canvas for a genuinely empty
group, gating, and configuration errors carrying step context. Engine imports
happen inside those tests so that block-only tests collect and run even when
the engine is unavailable.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v2.blocks import (
    CROP_BLOCK_NAME,
    HAS_BRIGHTNESS_BLOCK_NAME,
    INVERT_BLOCK_NAME,
    MOSAIC_BLOCK_NAME,
    NATIVE_BLOCK_NAMES,
    NATIVE_KIND_NAMES,
    CropBlock,
    HasBrightnessBlock,
    InvertBlock,
    MosaicBlock,
    is_boolean_payload,
    is_crop_summary_payload,
    is_image_payload,
    is_integer_payload,
    native_registry,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import ContractError

REGIONS = [[40, 40, 100, 100], [120, 10, 180, 70]]


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


class TestKindValidators:
    def test_image_validator_accepts_rgb_uint8(self) -> None:
        assert is_image_payload(_image(4, 5)) is True

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
    def test_image_validator_rejects_other_payloads(self, payload) -> None:
        assert is_image_payload(payload) is False

    def test_boolean_validator_rejects_numpy_and_int(self) -> None:
        assert is_boolean_payload(True) is True
        assert is_boolean_payload(np.bool_(True)) is False
        assert is_boolean_payload(1) is False

    def test_integer_validator_rejects_bool_and_float(self) -> None:
        assert is_integer_payload(3) is True
        assert is_integer_payload(True) is False
        assert is_integer_payload(3.0) is False

    def test_crop_summary_validator(self) -> None:
        assert is_crop_summary_payload(
            {"crop_count": 1, "image_height": 2, "image_width": 3}
        )
        assert not is_crop_summary_payload({"crop_count": 1})
        assert not is_crop_summary_payload([1, 2, 3])


class TestCropBlock:
    def test_ragged_counts_and_sparse_local_indices(self) -> None:
        block = CropBlock.from_config({"regions": REGIONS})
        results = [block.run(image=image) for image in _three_images()]

        counts = [len(result["crops"]) for result in results]
        assert counts == [2, 0, 1]
        assert results[0]["crops"].indices == ((0,), (1,))
        assert results[1]["crops"].indices == ()
        assert results[2]["crops"].indices == ((0,),)
        assert [result["summary"]["crop_count"] for result in results] == [2, 0, 1]

    def test_omitted_second_region_keeps_configured_index_of_third(self) -> None:
        regions = [[0, 0, 10, 10], [500, 500, 600, 600], [10, 10, 20, 20]]
        block = CropBlock.from_config({"regions": regions})
        result = block.run(image=_image(40, 40))

        assert result["crops"].indices == ((0,), (2,))
        assert result["summary"]["kept_regions"] == [0, 2]

    def test_clipping_and_pixel_equality(self) -> None:
        image = _image(50, 60)
        block = CropBlock.from_config(
            {"regions": [[-10, -10, 20, 30], [50, 40, 100, 100]]}
        )
        result = block.run(image=image)

        assert len(result["crops"]) == 2
        np.testing.assert_array_equal(result["crops"][0], image[0:30, 0:20])
        np.testing.assert_array_equal(result["crops"][1], image[40:50, 50:60])
        assert result["summary"]["crop_dimensions"] == [[30, 20], [10, 10]]

    def test_crops_are_copies_not_views(self) -> None:
        image = _image(50, 60)
        result = CropBlock.from_config({"regions": [[0, 0, 10, 10]]}).run(image=image)
        crop = result["crops"][0]
        crop[...] = 0

        assert image[0:10, 0:10].any()

    def test_zero_area_configured_rectangle_is_omitted(self) -> None:
        result = CropBlock.from_config({"regions": [[5, 5, 5, 20]]}).run(
            image=_image(30, 30)
        )

        assert len(result["crops"]) == 0
        assert isinstance(result["crops"], Batch)

    @pytest.mark.parametrize(
        "config, fragment",
        [
            ({}, "requires config key 'regions'"),
            ({"regions": "no"}, "must be a list"),
            ({"regions": [[0, 0, 10]]}, "regions[0]"),
            ({"regions": [[0, 0, 10, "a"]]}, "integers"),
            ({"regions": [[10, 0, 0, 10]]}, "x0 <= x1"),
            ({"regions": [], "padding": 2}, "unknown config keys ['padding']"),
        ],
    )
    def test_config_errors(self, config, fragment) -> None:
        with pytest.raises(ContractError, match=".*") as info:
            CropBlock.from_config(config)

        assert fragment in str(info.value)

    def test_rejects_non_image_payload(self) -> None:
        block = CropBlock.from_config({"regions": REGIONS})
        with pytest.raises(ContractError) as info:
            block.run(image=np.zeros((4, 4), dtype=np.uint8))

        assert "v2/crop input 'image'" in str(info.value)


class TestInvertBlock:
    def test_pixel_inversion_and_input_untouched(self) -> None:
        image = _image(8, 9)
        original = image.copy()
        result = InvertBlock.from_config({}).run(image=image)

        np.testing.assert_array_equal(result["image"], 255 - original)
        np.testing.assert_array_equal(image, original)
        assert result["image"].dtype == np.uint8

    def test_rejects_config(self) -> None:
        with pytest.raises(ContractError):
            InvertBlock.from_config({"strength": 2})


class TestMosaicBlock:
    def test_empty_group_yields_blank_canvas_and_zero(self) -> None:
        block = MosaicBlock.from_config({"tile_size": 16, "background": 7})
        result = block.run(images=Batch.empty())

        assert result["count"] == 0
        assert result["image"].shape == (16, 16, 3)
        assert int(result["image"].min()) == 7 and int(result["image"].max()) == 7

    def test_two_constant_images_fill_two_tiles(self) -> None:
        red = np.full((10, 20, 3), (200, 0, 0), dtype=np.uint8)
        blue = np.full((30, 5, 3), (0, 0, 200), dtype=np.uint8)
        block = MosaicBlock.from_config({"tile_size": 8})
        result = block.run(images=Batch.of([red, blue]))

        assert result["count"] == 2
        assert result["image"].shape == (8, 16, 3)
        np.testing.assert_array_equal(
            result["image"][:, :8], np.broadcast_to(red[0, 0], (8, 8, 3))
        )
        np.testing.assert_array_equal(
            result["image"][:, 8:], np.broadcast_to(blue[0, 0], (8, 8, 3))
        )

    def test_columns_config_controls_grid(self) -> None:
        images = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        result = MosaicBlock.from_config({"tile_size": 4, "columns": 1}).run(
            images=images
        )

        assert result["image"].shape == (12, 4, 3)
        assert [int(result["image"][row * 4, 0, 0]) for row in range(3)] == [10, 20, 30]

    @pytest.mark.parametrize(
        "config",
        [
            {"tile_size": 0},
            {"tile_size": True},
            {"columns": -1},
            {"background": 256},
            {"size": 3},
        ],
    )
    def test_config_errors(self, config) -> None:
        with pytest.raises(ContractError):
            MosaicBlock.from_config(config)

    def test_rejects_non_image_member(self) -> None:
        block = MosaicBlock.from_config({})
        with pytest.raises(ContractError) as info:
            block.run(images=[_image(4, 4), "nope"])

        assert "images[1]" in str(info.value)


class TestHasBrightnessBlock:
    def test_threshold_is_inclusive_and_returns_bool(self) -> None:
        block = HasBrightnessBlock.from_config({"minimum": 100})
        exact = np.full((3, 3, 3), 100, dtype=np.uint8)
        dark = np.full((3, 3, 3), 99, dtype=np.uint8)

        assert block.run(image=exact)["keep"] is True
        assert block.run(image=dark)["keep"] is False

    @pytest.mark.parametrize(
        "config", [{}, {"minimum": 300}, {"minimum": "5"}, {"minimum": True}]
    )
    def test_config_errors(self, config) -> None:
        with pytest.raises(ContractError):
            HasBrightnessBlock.from_config(config)


class TestNativeRegistry:
    def test_registers_declared_kinds_and_blocks(self) -> None:
        registry = native_registry()

        assert registry.kind_names == NATIVE_KIND_NAMES
        assert registry.block_names == NATIVE_BLOCK_NAMES
        assert set(NATIVE_BLOCK_NAMES) == {
            CROP_BLOCK_NAME,
            INVERT_BLOCK_NAME,
            MOSAIC_BLOCK_NAME,
            HAS_BRIGHTNESS_BLOCK_NAME,
        }

    def test_each_call_returns_an_independent_registry(self) -> None:
        first = native_registry()
        second = native_registry()
        first.register_kind("number")

        assert not second.has_kind("number")

    def test_factories_produce_runnable_instances(self) -> None:
        registry = native_registry()
        registration = registry.get_block(CROP_BLOCK_NAME)
        block = registration.factory({"regions": REGIONS})

        assert len(block.run(image=_image(120, 200))["crops"]) == 2
        assert registration.contract.reference == "image"
        assert registration.contract.outputs["crops"].transform == "append"
        assert registration.contract.outputs["summary"].transform == "preserve"

    def test_kind_validation_through_registry(self) -> None:
        registry = native_registry()
        registry.validate("image", _image(2, 2))
        with pytest.raises(ContractError):
            registry.validate("image", "not an image")

    def test_snapshot_isolated_from_later_registration(self) -> None:
        registry = native_registry()
        snapshot = registry.snapshot()
        registry.register_kind("later")

        assert not snapshot.has_kind("later")


class TestCatalogueIsolation:
    def test_generic_core_import_does_not_load_native_blocks(self) -> None:
        script = (
            "import sys\n"
            "import roboflow_workflows.execution_engine.v2\n"
            "import roboflow_workflows.execution_engine.v2.data\n"
            "import roboflow_workflows.execution_engine.v2.contracts\n"
            "loaded = [m for m in sys.modules if m.startswith("
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
        loader_source = (
            Path(__file__).resolve().parents[4]
            / "roboflow_workflows"
            / "core_steps"
            / "loader.py"
        ).read_text()

        assert "execution_engine.v2" not in loader_source


def _definition(steps: list, outputs: list) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [
            {
                "name": "images",
                "kind": "image",
                "axes": [{"id": "samples", "kind": "sample", "stationary": True}],
            }
        ],
        "steps": steps,
        "outputs": outputs,
    }

    return definition


CROP_STEP = {
    "name": "crop",
    "type": "v2/crop",
    "inputs": {"image": "$inputs.images"},
    "config": {"regions": REGIONS},
}


class TestCompiledNativeWorkflows:
    def test_crop_invert_mosaic_with_ragged_children(self) -> None:
        from roboflow_workflows.execution_engine.v2 import compile_workflow

        definition = _definition(
            steps=[
                CROP_STEP,
                {
                    "name": "invert",
                    "type": "v2/invert",
                    "inputs": {"image": "$steps.crop.crops"},
                    "config": {},
                },
                {
                    "name": "mosaic",
                    "type": "v2/mosaic",
                    "inputs": {"images": "$steps.invert.image"},
                    "config": {"tile_size": 16, "background": 9},
                },
            ],
            outputs=[
                {"name": "crops", "selector": "$steps.crop.crops"},
                {"name": "summary", "selector": "$steps.crop.summary"},
                {"name": "mosaic", "selector": "$steps.mosaic.image"},
                {"name": "count", "selector": "$steps.mosaic.count"},
            ],
        )
        images = _three_images()
        plan = compile_workflow(definition, registry=native_registry())
        result = plan.run(inputs={"images": Batch.of(images)})
        outputs = result.outputs

        crops = outputs.data["crops"]
        assert [len(group) for group in crops] == [2, 0, 1]
        assert crops[0].indices == ((0, 0), (0, 1))
        assert crops[1].indices == ()
        assert crops[1].parent_index == (1,)
        assert crops[2].indices == ((2, 0),)
        assert [axis.kind for axis in outputs.layout["crops"].axes] == [
            "sample",
            "dynamic_nesting",
        ]
        assert [axis.kind for axis in outputs.layout["summary"].axes] == ["sample"]
        assert [axis.kind for axis in outputs.layout["mosaic"].axes] == ["sample"]

        direct = CropBlock.from_config({"regions": REGIONS}).run(image=images[0])
        np.testing.assert_array_equal(crops[0][1], direct["crops"][1])
        assert [s["crop_count"] for s in outputs.data["summary"]] == [2, 0, 1]

        assert list(outputs.data["count"]) == [2, 0, 1]
        blank = outputs.data["mosaic"][1]
        assert blank.shape == (16, 16, 3)
        assert int(blank.min()) == 9 and int(blank.max()) == 9
        assert all(status == "complete" for status in result.statuses.values())

    def test_gate_filters_children_without_renumbering(self) -> None:
        from roboflow_workflows.execution_engine.v2 import compile_workflow

        bright = np.full((120, 200, 3), 230, dtype=np.uint8)
        bright[10:70, 120:180] = 5
        dark = np.full((110, 110, 3), 5, dtype=np.uint8)
        definition = _definition(
            steps=[
                CROP_STEP,
                {
                    "name": "keep",
                    "type": "v2/has_brightness",
                    "inputs": {"image": "$steps.crop.crops"},
                    "config": {"minimum": 100},
                },
                {
                    "name": "invert",
                    "type": "v2/invert",
                    "inputs": {"image": "$steps.crop.crops"},
                    "config": {},
                    "when": "$steps.keep.keep",
                },
            ],
            outputs=[
                {"name": "keep", "selector": "$steps.keep.keep"},
                {"name": "inverted", "selector": "$steps.invert.image"},
            ],
        )
        plan = compile_workflow(definition, registry=native_registry())
        result = plan.run(inputs={"images": Batch.of([bright, _image(32, 32), dark])})

        keep = result.outputs.data["keep"]
        assert list(keep[0]) == [True, False]
        assert list(keep[2]) == [False]
        inverted = result.outputs.data["inverted"]
        assert inverted[0].indices == ((0, 0),)
        np.testing.assert_array_equal(inverted[0][0], 255 - bright[40:100, 40:100])

    def test_config_error_names_the_step(self) -> None:
        from roboflow_workflows.execution_engine.v2 import compile_workflow

        definition = _definition(
            steps=[
                {
                    "name": "bad_crop",
                    "type": "v2/crop",
                    "inputs": {"image": "$inputs.images"},
                    "config": {"regions": [[0, 0, 10]]},
                }
            ],
            outputs=[{"name": "crops", "selector": "$steps.bad_crop.crops"}],
        )
        with pytest.raises(ValueError) as info:
            compile_workflow(definition, registry=native_registry())

        assert "bad_crop" in str(info.value)
        assert "regions[0]" in str(info.value) or "regions[0]" in str(
            info.value.__cause__
        )


def test_contiguous_and_overlapping_crops_have_independent_pixel_storage() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    original = image.copy()
    block = CropBlock.from_config(
        {"regions": [[0, 0, 5, 3], [20, 20, 21, 21], [0, 1, 5, 3], [0, 0, 3, 2]]}
    )

    crops = block.run(image=image)["crops"]

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
    np.testing.assert_array_equal(crops[2], original[:2, :3])
