"""Compiled workflows that generate, crop, read, filter and draw codes.

These run the real compiler, session and executor; blocks are never called
directly. Every code is generated in the test, so expected texts are exact.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
import qrcode
import torch
import zxingcpp
from roboflow_workflows.v2.blocks.codes.barcode_detector.v1 import BarcodeDetectorBlock
from roboflow_workflows.v2.blocks.codes.qr_code_detector.v1 import QRCodeDetectorBlock
from roboflow_workflows.v2.blocks.codes.qr_code_generator.v1 import QRCodeGeneratorBlock
from roboflow_workflows.v2.blocks.image.primitives.static_crop.v1 import StaticCropBlock
from roboflow_workflows.v2.blocks.transforms.detections_filter.v1 import DetectionsFilterBlock
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import BoundingBoxVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import LabelVisualizationBlock
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions

CATALOGUE = Catalogue(
    [
        QRCodeGeneratorBlock,
        QRCodeDetectorBlock,
        BarcodeDetectorBlock,
        StaticCropBlock,
        DetectionsFilterBlock,
        BoundingBoxVisualizationBlock,
        LabelVisualizationBlock,
    ]
)
IMAGES_INPUT = {"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _texts(predictions) -> List[str]:
    texts = [row["data"] for row in predictions.bboxes_metadata]

    return texts


def _qr(text: str, module: int = 5) -> np.ndarray:
    code = qrcode.QRCode(box_size=module, border=4)
    code.add_data(text)
    code.make(fit=True)
    gray = np.asarray(code.make_image().convert("L"), dtype=np.uint8)

    return gray


def _rgb_canvas(height: int, width: int, *codes) -> np.ndarray:
    canvas = np.full((height, width), 255, dtype=np.uint8)
    for code, x, y in codes:
        canvas[y : y + code.shape[0], x : x + code.shape[1]] = code
    rgb = np.stack([canvas] * 3, axis=-1)

    return rgb


def _generate_and_read_definition() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "text", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "border", "kind": ["integer"]},
        ],
        "steps": [
            {
                "type": "roboflow_core/qr_code_generator@v1",
                "name": "generate",
                "text": "$inputs.text",
                "border": "$inputs.border",
                "fill_color": "navy",
            },
            {
                "type": "roboflow_core/qr_code_detector@v1",
                "name": "qr",
                "image": "$steps.generate.qr_code",
            },
            {
                "type": "roboflow_core/barcode_detector@v1",
                "name": "barcodes",
                "image": "$steps.generate.qr_code",
            },
        ],
        "outputs": [
            _output("qr_code", "$steps.generate.qr_code"),
            _output("qr", "$steps.qr.predictions"),
            _output("barcodes", "$steps.barcodes.predictions"),
        ],
    }

    return definition


def _read_filter_draw_steps(image: str, *, detector: str, property_name: str):
    """Steps: detector -> keep rows whose text equals ``$inputs.wanted`` -> draw."""
    text_filter = {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": property_name,
                            }
                        ],
                    },
                    "comparator": {"type": "=="},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "wanted",
                    },
                }
            ],
        },
    }
    steps = [
        {"type": detector, "name": "read", "image": image},
        {
            "type": "roboflow_core/detections_filter@v1",
            "name": "keep",
            "predictions": "$steps.read.predictions",
            "operations": [text_filter],
            "operations_parameters": {"wanted": "$inputs.wanted"},
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "boxes",
            "image": image,
            "predictions": "$steps.keep.predictions",
        },
        {
            "type": "roboflow_core/label_visualization@v2",
            "name": "labels",
            "image": "$steps.boxes.image",
            "predictions": "$steps.keep.predictions",
        },
    ]

    return steps


def _changed(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    changed = (before != after).any(dim=0)

    return changed


@pytest.mark.parametrize(
    "text, wanted, kept", [("LOT-42", "LOT-42", 1), ("LOT-42", "LOT-7", 0)]
)
def test_generated_code_is_read_filtered_by_text_and_drawn(text, wanted, kept) -> None:
    # given: the generator's defaults, so border 4.
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "text", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "wanted", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": "roboflow_core/qr_code_generator@v1",
                "name": "generate",
                "text": "$inputs.text",
            },
            *_read_filter_draw_steps(
                "$steps.generate.qr_code",
                detector="roboflow_core/qr_code_detector@v1",
                property_name="decoded_text",
            ),
        ],
        "outputs": [
            _output("qr_code", "$steps.generate.qr_code"),
            _output("read", "$steps.read.predictions"),
            _output("kept", "$steps.keep.predictions"),
            _output("drawn", "$steps.labels.image"),
        ],
    }
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    # when
    try:
        row = session.run({"text": text, "wanted": wanted}).rows()[0]
    finally:
        session.close()

    # then
    assert _texts(row["read"]) == [text]
    assert _texts(row["kept"]) == [text] * kept
    changed = _changed(row["qr_code"].tensor_image, row["drawn"].tensor_image)
    assert bool(changed.any()) == bool(kept)
    assert row["drawn"].image_id == row["qr_code"].image_id


@pytest.mark.parametrize(
    "detector, property_name",
    [
        ("roboflow_core/qr_code_detector@v1", "decoded_text"),
        ("roboflow_core/barcode_detector@v1", "data"),
    ],
)
def test_filter_drops_unwanted_codes_and_painters_draw_only_kept_rows(
    detector, property_name
) -> None:
    # given: three codes, two with the wanted text.
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "wanted", "kind": ["string"]},
        ],
        "steps": _read_filter_draw_steps(
            "$inputs.image", detector=detector, property_name=property_name
        ),
        "outputs": [
            _output("read", "$steps.read.predictions"),
            _output("kept", "$steps.keep.predictions"),
            _output("drawn", "$steps.labels.image"),
        ],
    }
    codes = [
        (_qr("LOT-42"), 20, 120),
        (_qr("LOT-99"), 300, 120),
        (_qr("LOT-42"), 580, 120),
    ]
    image = _rgb_canvas(360, 860, *codes)
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    # when
    try:
        row = session.run({"image": image, "wanted": "LOT-42"}).rows()[0]
    finally:
        session.close()

    # then: kept rows keep their boxes, metadata, frame and row identity.
    read, kept = row["read"], row["kept"]
    read_boxes = dict(zip(map(tuple, read.xyxy.tolist()), _texts(read)))
    assert sorted(_texts(read)) == ["LOT-42", "LOT-42", "LOT-99"]
    assert _texts(kept) == ["LOT-42", "LOT-42"]
    assert all(read_boxes[tuple(box)] == "LOT-42" for box in kept.xyxy.tolist())
    assert kept.image_metadata["class_names"] == read.image_metadata["class_names"]
    assert frame_of(kept) == frame_of(read)
    assert row_record(kept).namespace == row_record(read).namespace
    assert len(set(row_record(kept).host_ids())) == 2
    # Painters change pixels at kept codes only; the LOT-99 code stays intact.
    changed = _changed(
        torch.from_numpy(image).permute(2, 0, 1), row["drawn"].tensor_image
    )
    x0, y0, x1, y1 = next(box for box, text in read_boxes.items() if text == "LOT-99")
    assert not changed[int(y0) - 5 : int(y1) + 5, int(x0) - 5 : int(x1) + 5].any()
    for x0, y0, x1, y1 in kept.xyxy.tolist():
        assert changed[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1].any()


@pytest.mark.parametrize("border", [2, 4])
@pytest.mark.parametrize("mode", ["run", "phases"])
def test_generated_code_is_read_back_by_both_detectors(mode, border) -> None:
    # given
    plan = compile_workflow(
        _generate_and_read_definition(),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session()

    # when
    try:
        row = session.run({"text": "https://roboflow.com", "border": border}).rows()[0]
    finally:
        session.close()

    # then
    image = row["qr_code"]
    for name in ("qr", "barcodes"):
        predictions = row[name]
        assert _texts(predictions) == ["https://roboflow.com"]
        assert frame_of(predictions).image_id == image.image_id
        x0, y0, x1, y1 = predictions.xyxy[0].tolist()
        # ``border`` modules of 10 px quiet zone on every side.
        margin = 10 * border
        assert (x0, y0) == pytest.approx((margin, margin), abs=3)
        assert x1 == pytest.approx(image.width - margin, abs=3)
        assert y1 == pytest.approx(image.height - margin, abs=3)


def test_pipelined_runs_read_their_own_text() -> None:
    # given
    session = compile_workflow(
        _generate_and_read_definition(), catalogue=CATALOGUE
    ).create_session()
    texts = [f"pulse-{index}" for index in range(6)]

    # when
    try:
        with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipe:
            futures = [pipe.submit({"text": text, "border": 2}) for text in texts]
        rows = [future.result().rows()[0] for future in futures]
    finally:
        session.close()

    # then
    assert [_texts(row["qr"]) for row in rows] == [[text] for text in texts]
    assert len({row["qr_code"].image_id for row in rows}) == len(texts)


def test_ragged_batch_keeps_one_aligned_result_per_image() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "roboflow_core/barcode_detector@v1",
                "name": "codes",
                "image": "$inputs.images",
            }
        ],
        "outputs": [_output("predictions", "$steps.codes.predictions")],
    }
    code_128 = np.asarray(
        zxingcpp.write_barcode(
            zxingcpp.BarcodeFormat.Code128,
            "LOT-42",
            width=300,
            height=80,
            quiet_zone=20,
        )
    )
    images = [
        _rgb_canvas(260, 520, (_qr("left"), 10, 10), (_qr("right"), 300, 60)),
        _rgb_canvas(7, 3),
        _rgb_canvas(200, 400, (code_128, 30, 40)),
    ]
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    # when
    try:
        rows = session.run({"images": images}).rows()
    finally:
        session.close()

    # then
    assert [sorted(_texts(row["predictions"])) for row in rows] == [
        ["left", "right"],
        [],
        ["LOT-42"],
    ]
    assert [frame_of(row["predictions"]).size_hw for row in rows] == [
        (260, 520),
        (7, 3),
        (200, 400),
    ]


def test_codes_read_in_static_crops_map_back_to_the_input_image() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/static_crop@v1",
                "name": "crop",
                "image": "$inputs.image",
                "regions": [[0, 0, 250, 250], [250, 0, 500, 250]],
            },
            {
                "type": "roboflow_core/qr_code_detector@v1",
                "name": "qr",
                "image": "$steps.crop.crops",
            },
        ],
        "outputs": [
            _output("in_input", "$steps.qr.predictions"),
            {
                **_output("in_crop", "$steps.qr.predictions"),
                "coordinates_system": "own",
            },
        ],
    }
    left, right = _qr("west"), _qr("east")
    image = _rgb_canvas(250, 500, (left, 20, 30), (right, 290, 60))
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    # when
    try:
        row = session.run({"image": image}).rows()[0]
    finally:
        session.close()

    # then: outputs default to input-image coordinates, as in V1.
    in_crop, in_input = row["in_crop"], row["in_input"]
    assert [_texts(predictions) for predictions in in_crop] == [["west"], ["east"]]
    crop_x, crop_y, _, _ = in_crop[1].xyxy[0].tolist()
    assert (crop_x, crop_y) == pytest.approx((290 - 250 + 20, 60 + 20), abs=3)
    assert frame_of(in_crop[1]).root.map_xy(crop_x, crop_y) == pytest.approx(
        (290 + 20, 60 + 20), abs=3
    )
    assert in_input[1].xyxy[0].tolist()[:2] == pytest.approx((310, 80), abs=3)
    assert _texts(in_input[1]) == ["east"]


def test_selected_border_is_checked_at_run_time() -> None:
    # given
    session = compile_workflow(
        _generate_and_read_definition(), catalogue=CATALOGUE
    ).create_session()

    # when / then
    try:
        with pytest.raises(StepExecutionError, match=r"border must be in \[0, 100\]"):
            session.run({"text": "A", "border": 500})
    finally:
        session.close()


@pytest.mark.parametrize(
    "field, value",
    [
        ("border", 500),
        ("border", -1),
        ("error_correct", "Maximum"),
    ],
)
def test_invalid_written_parameters_fail_compilation(field, value) -> None:
    # given
    definition = _generate_and_read_definition()
    definition["steps"][0][field] = value

    # when / then
    with pytest.raises(ParamsValidationError, match=field):
        compile_workflow(definition, catalogue=CATALOGUE)
