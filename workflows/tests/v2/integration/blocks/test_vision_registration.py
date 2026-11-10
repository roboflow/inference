"""Exercise default public discovery and CPU segmentation/drawing composition."""

import json

import pytest
import torch
from roboflow_workflows.v2 import blocks
from roboflow_workflows.v2.blocks import image, models
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.visualization import annotations
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.integration.blocks.visualization.annotations import (
    test_counter_drawing_composition as counters,
)
from tests.v2.support.models.fake_sam_models import FakeSam2, FakeSam3

_ADDED = (
    (models, "SegmentAnything2Block", "roboflow_core/segment_anything@v1"),
    (models, "SegmentAnything2VideoBlock", "roboflow_core/segment_anything_2_video@v1"),
    (models, "SegmentAnything3Block", "roboflow_core/sam3@v3"),
    (models, "SegmentAnything3InteractiveBlock", "roboflow_core/sam3_interactive@v1"),
    (models, "SegmentAnything3VideoBlock", "roboflow_core/sam3_video@v1"),
    (image, "MaskEdgeSnapBlock", "roboflow_core/mask_edge_snap@v1"),
    (
        annotations,
        "HeatmapVisualizationBlock",
        "roboflow_core/heatmap_visualization@v1",
    ),
    (annotations, "TraceVisualizationBlock", "roboflow_core/trace_visualization@v1"),
    (
        annotations,
        "PolygonZoneVisualizationBlock",
        "roboflow_core/polygon_zone_visualization@v1",
    ),
    (
        annotations,
        "LineCounterVisualizationBlock",
        "roboflow_core/line_counter_visualization@v1",
    ),
    (
        annotations,
        "ReferencePathVisualizationBlock",
        "roboflow_core/reference_path_visualization@v1",
    ),
)


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
def test_default_discovery_public_exports_and_schemas(route):
    if route == "create":
        catalogue = blocks.create_catalogue()
    elif route == "plugin":
        catalogue = Catalogue.from_modules(["roboflow_workflows.v2.blocks"])
    else:
        catalogue = Catalogue.from_modules(
            ["roboflow_workflows.v2.blocks"], attribute="create_catalogue"
        )

    described = catalogue.describe()
    json.dumps(described)
    schemas = {row["type"]: row["params_schema"] for row in described["blocks"]}
    for package, name, identity in _ADDED:
        block = getattr(package, name)
        assert block.type == identity
        assert catalogue.block_types.count(identity) == 1
        assert package.__all__.count(name) == blocks.__all__.count(name) == 1
        assert getattr(blocks, name) is block
        spec = spec_of(block)
        assert catalogue.entry(identity).spec is spec
        for alias in spec.identities:
            assert catalogue.entry(alias).spec is spec
        assert schemas[identity] == spec.params_model.model_json_schema()
        assert "image" in schemas[identity]["properties"]

    # Keep both public label versions and Action/stable19 additions discoverable.
    assert blocks.LabelVisualizationBlock.type == "roboflow_core/label_visualization@v2"
    for identity in (
        "roboflow_core/roboflow_action_recognition_model@v1",
        "roboflow_core/cosmos3_edge@v1",
        "roboflow_core/florence_2@v2",
        "roboflow_core/google_vision_ocr@v1",
        "roboflow_core/moondream2@v1",
        "roboflow_core/qwen_vlm@v4",
        "roboflow_core/visual_search@v1",
        "roboflow_core/visual_search_classifier@v1",
    ):
        assert catalogue.block_types.count(identity) == 1
    for identity in (
        "v2/segment_anything_3d@v1",
        "v2/sam3_3d@v1",
        "v2/hosted_seg_preview@v1",
    ):
        assert catalogue.find(identity) is None


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("backend", ["sam2", "sam3"])
def test_default_segmentation_snap_and_drawing_preserve_pixels_and_identity(
    mode, backend
):
    model = (
        FakeSam2()
        if backend == "sam2"
        else FakeSam3(table={"part": [((2, 3, 8, 9), 0.9)]})
    )
    loaded = []

    def load(request):
        loaded.append(request.model_id)
        return model

    step = {
        "type": (
            "roboflow_core/segment_anything@v1"
            if backend == "sam2"
            else "roboflow_core/sam3@v3"
        ),
        "name": "sam",
        "image": "$inputs.image",
    }
    if backend == "sam3":
        step["class_names"] = ["part"]

    drawing = {
        "type": "roboflow_core/mask_visualization@v1",
        "image": "$inputs.image",
        "color_palette": "CUSTOM",
        "custom_colors": ["#ff0000"],
        "opacity": 1.0,
    }
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            step,
            {
                "type": "roboflow_core/mask_edge_snap@v1",
                "name": "snap",
                "image": "$inputs.image",
                "segmentation": "$steps.sam.segmentation",
                "dilation_iterations": 0,
                "min_contour_area": 0.0,
            },
            {**drawing, "name": "draw_view", "predictions": "$steps.sam.segmentation"},
            {
                **drawing,
                "name": "draw_dense",
                "predictions": "$steps.snap.refined_segmentation",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in [
                ("masks", "$steps.sam.predictions"),
                ("refined", "$steps.snap.refined_segmentation"),
                ("view", "$steps.draw_view.image"),
                ("dense", "$steps.draw_dense.image"),
            ]
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=blocks.create_catalogue(),
        options=CompileOptions(block_execution=mode),
    )
    assert loaded == []
    pixels = torch.zeros((3, 20, 30), dtype=torch.uint8)
    source = ImageData.from_tensor(pixels, image_id="segmentation-drawing")
    with plan.create_session({"models": ModelCache(load)}) as session:
        (row,) = session.run({"image": [source]}).rows()

    assert len(loaded) == 1
    assert torch.equal(row["masks"].mask, row["refined"].mask)
    mask = row["masks"].mask.any(dim=0)
    assert mask.any() and not mask.all()
    expected = torch.zeros_like(pixels)
    expected[0, mask] = 255
    for name in ("view", "dense"):
        drawn = row[name]
        assert torch.equal(drawn.tensor_image, expected)
        assert drawn.image_id == source.image_id
        assert drawn.root == source.root and drawn.parent == source.parent
        assert drawn.tensor_image.device == source.tensor_image.device
    assert torch.count_nonzero(source.tensor_image) == 0


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_default_drawings_compose_with_explicit_current_real_producers(
    mode, monkeypatch
):
    original = counters.compile_workflow
    expected = blocks.create_catalogue().block_types

    def compile_default(definition, *, catalogue, **kwargs):
        # The tail cohort registers the current real producers by default.
        assert catalogue.block_types == expected
        assert catalogue.entry(
            "roboflow_core/line_counter_visualization@v1"
        ).spec is spec_of(annotations.LineCounterVisualizationBlock)
        return original(definition, catalogue=catalogue, **kwargs)

    monkeypatch.setattr(counters, "compile_workflow", compile_default)
    counters.test_real_counter_outputs_draw_correct_labels_and_shared_zone_across_sources(
        mode
    )
