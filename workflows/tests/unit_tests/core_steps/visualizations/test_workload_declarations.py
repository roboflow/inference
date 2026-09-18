"""Workload declarations of the visualization blocks.

Every visualization block draws, so ``VISUALIZATION`` is expected on all of
them - but the category must not be a blanket label: the blocks that also
compose several images into one declare ``IMAGE_COMPOSITION`` as well, and the
two blocks that accumulate per-video state declare that state's caveat.
"""

from typing import List

from roboflow_workflows.core_steps.visualizations.bounding_box.v1 import (
    BoundingBoxManifest,
)
from roboflow_workflows.core_steps.visualizations.grid.v1 import (
    GridVisualizationManifest,
)
from roboflow_workflows.core_steps.visualizations.heatmap.v1 import HeatmapManifest
from roboflow_workflows.core_steps.visualizations.trace.v1 import TraceManifest
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    get_manifest_type_identifiers,
    load_workflow_blocks,
)

STATEFUL_VISUALIZATION_CODES = [
    "stateful_video_state_resets_on_stateless_http",
    "temporal_block_no_benefit_on_still_image",
]


def _visualization_blocks():
    for block in load_workflow_blocks():
        manifest_class = block.manifest_class
        schema = manifest_class.model_json_schema()
        if schema.get("block_type") != "visualization":
            continue
        identifiers = get_manifest_type_identifiers(
            block_schema=schema,
            block_source=block.block_source,
            block_identifier=block.identifier,
        )
        yield identifiers[0], manifest_class


def _instance(block_type: str, manifest_class):
    values = {"type": block_type, "name": "visualization_step"}
    for field_name, field in manifest_class.model_fields.items():
        if field_name not in values and field.is_required():
            values[field_name] = None
    return manifest_class.model_construct(**values)


def test_every_visualization_block_declares_visualization() -> None:
    missing: List[str] = []
    for block_type, manifest_class in _visualization_blocks():
        operations = _instance(block_type, manifest_class).discover_work_operations()
        if WorkOperation.VISUALIZATION not in operations:
            missing.append(block_type)
    assert not missing, missing


def test_visualization_blocks_do_not_all_declare_the_same_thing() -> None:
    declared = {
        tuple(
            operation.value
            for operation in _instance(
                block_type, manifest_class
            ).discover_work_operations()
        )
        for block_type, manifest_class in _visualization_blocks()
    }
    assert len(declared) >= 2, declared


def test_a_plain_annotator_declares_only_visualization() -> None:
    manifest = BoundingBoxManifest(
        type="roboflow_core/bounding_box_visualization@v1",
        name="boxes",
        image="$inputs.image",
        predictions="$steps.model.predictions",
    )
    assert manifest.discover_work_operations() == [WorkOperation.VISUALIZATION]
    assert manifest.discover_portable_restrictions() == []


def test_the_grid_block_also_declares_composition() -> None:
    manifest = GridVisualizationManifest(
        type="roboflow_core/grid_visualization@v1",
        name="grid",
        images="$steps.crop.crops",
    )
    assert manifest.discover_work_operations() == [
        WorkOperation.VISUALIZATION,
        WorkOperation.IMAGE_COMPOSITION,
    ]


def test_heatmap_and_trace_declare_their_cross_frame_state() -> None:
    heatmap = HeatmapManifest(
        type="roboflow_core/heatmap_visualization@v1",
        name="heatmap",
        image="$inputs.image",
        predictions="$steps.model.predictions",
    )
    trace = TraceManifest(
        type="roboflow_core/trace_visualization@v1",
        name="trace",
        image="$inputs.image",
        predictions="$steps.tracker.tracked_detections",
    )
    for manifest in (heatmap, trace):
        assert manifest.discover_work_operations() == [
            WorkOperation.VISUALIZATION,
            WorkOperation.TEMPORAL_BUFFERING,
        ]
        assert [
            restriction.code
            for restriction in manifest.discover_portable_restrictions()
        ] == STATEFUL_VISUALIZATION_CODES
