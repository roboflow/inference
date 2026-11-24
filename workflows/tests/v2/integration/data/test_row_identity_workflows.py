"""Row identity through compiled workflows: select, join, convert, serialize.

A fake detector mints one identity per result, as model blocks do. Every
root yields the same boxes and local row ids, so only the namespace keeps
results of different roots apart. A join block matches selected rows back to
the original rows by row key, never by engine index or geometry.
"""

import json

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.image.primitives.resize.v1 import ResizeBlock
from roboflow_workflows.v2.blocks.image.primitives.static_crop.v1 import StaticCropBlock
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    NATIVE_KINDS,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    require_row_record,
    row_record,
    with_row_identity,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)

from inference_models.models.base.object_detection import Detections

LABELS = ("A", "B", "C")
# Selection applied before the join: filter, sort, repeat and empty.
SCENARIOS = {
    "filter": [1],
    "sort": [2, 0, 1],
    "repeat": [1, 1],
    "empty": [],
}


class _FakeDetector(Block):
    type = "row_identity_test/fake_detector"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to detect on.")

    def run(self, *, image):
        detections = Detections(
            xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]]).repeat(3, 1)
            + torch.arange(3.0)[:, None],
            class_id=torch.arange(3),
            confidence=torch.full((3,), 0.5),
            image_metadata=FrameRef.from_image(image).prediction_metadata(),
            bboxes_metadata=[{"label": label} for label in LABELS],
        )
        # G1 recipe: mint once, as the last metadata step of the result.
        return {"predictions": with_row_identity(detections)}


def _select_block(scenario):
    class _Select(Block):
        type = f"row_identity_test/select_{scenario}"
        outputs = {
            "predictions": Output(
                OBJECT_DETECTION_PREDICTION_KIND, source="predictions"
            )
        }

        class Params(BlockParams):
            predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
                description="Predictions to select from."
            )

        def run(self, *, predictions):
            selected = select_predictions(predictions, indices=SCENARIOS[scenario])
            return {"predictions": selected}

    return _Select


class _JoinToOriginal(Block):
    """Task-local stand-in for a keyed consumer such as class replacement."""

    type = "row_identity_test/join_to_original"
    outputs = {"labels": Output(LIST_OF_VALUES_KIND, source="original")}

    class Params(BlockParams):
        original: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Target rows."
        )
        selected: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Rows chosen from the targets."
        )

    def run(self, *, original, selected):
        targets = require_row_record(original, consumer=self.type)
        found = {}
        if len(selected):
            chosen = require_row_record(selected, consumer=self.type)
            assert chosen.namespace == targets.namespace
            for row_id, row in zip(chosen.host_ids(), selected.bboxes_metadata):
                found.setdefault(row_id, set()).add(row["label"])
        labels = [
            "|".join(sorted(found.get(row_id, {"fallback"})))
            for row_id in targets.host_ids()
        ]
        return {"labels": labels}


def _plan(scenario):
    select = _select_block(scenario)
    catalogue = Catalogue(
        [StaticCropBlock, ResizeBlock, _FakeDetector, select, _JoinToOriginal],
        kinds=[IMAGE_KIND, *NATIVE_KINDS, LIST_OF_VALUES_KIND],
    )
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": [
            {
                "type": "roboflow_core/static_crop@v1",
                "name": "crop",
                "image": "$inputs.images",
                "regions": [[4, 2, 24, 12]],
            },
            {
                "type": "roboflow_core/resize@v1",
                "name": "resize",
                "image": "$steps.crop.crops",
                "width": 10,
                "height": 5,
            },
            {
                "type": _FakeDetector.type,
                "name": "detect",
                "image": "$steps.resize.image",
            },
            {
                "type": select.type,
                "name": "select",
                "predictions": "$steps.detect.predictions",
            },
            {
                "type": _JoinToOriginal.type,
                "name": "join",
                "original": "$steps.detect.predictions",
                "selected": "$steps.select.predictions",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "labels", "selector": "$steps.join.labels"},
            {
                "type": "JsonField",
                "name": "own",
                "selector": "$steps.select.predictions",
                "coordinates_system": "own",
            },
            {
                "type": "JsonField",
                "name": "root",
                "selector": "$steps.select.predictions",
            },
        ],
    }
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def _roots():
    return [
        ImageData.from_tensor(
            torch.zeros((3, 30, 40), dtype=torch.uint8), image_id=name
        )
        for name in ("first", "second")
    ]


EXPECTED_LABELS = {
    "filter": ["fallback", "B", "fallback"],
    "sort": ["A", "B", "C"],
    "repeat": ["fallback", "B", "fallback"],
    "empty": ["fallback", "fallback", "fallback"],
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_selected_rows_join_back_to_their_original_rows(scenario):
    rows = _plan(scenario).create_session().run({"images": _roots()}).rows()

    for row in rows:
        (labels,) = row["labels"]
        assert labels == EXPECTED_LABELS[scenario]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_root_conversion_keeps_the_record_and_namespaces_stay_apart(scenario):
    rows = _plan(scenario).create_session().run({"images": _roots()}).rows()

    namespaces = set()
    for row in rows:
        (own,), (root,) = row["own"], row["root"]
        own_record, root_record = row_record(own), row_record(root)
        assert own_record.host_ids() == SCENARIOS[scenario]
        assert root_record.host_ids() == SCENARIOS[scenario]
        assert root_record.namespace == own_record.namespace
        # local (x, y) -> root (2x + 4, 2y + 2): geometry changed, identity kept.
        if len(own):
            assert torch.equal(root.xyxy, own.xyxy * 2 + torch.tensor([4, 2, 4, 2]))
        namespaces.add(own_record.namespace)

    assert len(namespaces) == 2


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_serialized_outputs_restore_the_record(scenario):
    result = _plan(scenario).create_session().run({"images": _roots()})
    own = result.rows()[0]["own"][0]

    wire = json.loads(json.dumps(result.rows(serialize=True)[0]["own"][0]))
    restored = OBJECT_DETECTION_PREDICTION_KIND.deserialize(wire)

    assert row_record(restored).namespace == row_record(own).namespace
    assert row_record(restored).host_ids() == SCENARIOS[scenario]
