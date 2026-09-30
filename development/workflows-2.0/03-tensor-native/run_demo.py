"""Run nested geometry, the native carrier gallery, and useful failure cases."""

import html
import json
from pathlib import Path
from typing import Any, Dict

import click
import torch
from author_blocks import create_demo_catalogue
from fixtures import (
    EXPECTED_ROOT_BOX,
    LOCAL_BOX,
    make_native_fixtures,
    make_root_pixels,
)
from PIL import Image, ImageDraw
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.predictions import NATIVE_KINDS
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

DEMO_DIR = Path(__file__).resolve().parent


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def _raw(result, name: str):
    (entry,) = result.selections[name].values()
    return result.outputs.data[entry]


def _save_overlay(image: ImageData, predictions: Detections, path: Path) -> None:
    # Pixel and box export is deliberately confined to visualization.
    pixels = image.tensor_image.detach().cpu().permute(1, 2, 0).numpy()
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    for box in predictions.xyxy.detach().cpu().tolist():
        draw.rectangle(box, outline=(255, 40, 40), width=2)
    canvas.save(path)


def _passthrough(kind: str, *, coordinates: str = "own") -> Dict[str, Any]:
    return {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": [kind]}],
        "steps": [],
        "outputs": [
            {
                "type": "JsonField",
                "name": "value",
                "selector": "$inputs.value",
                "coordinates_system": coordinates,
            }
        ],
    }


def _tensor_summary(payload: Any) -> list:
    if isinstance(payload, torch.Tensor):
        tensors = [payload]
    elif isinstance(payload, (Detections, InstanceDetections)):
        tensors = [payload.xyxy, payload.class_id, payload.confidence]
        if isinstance(payload, InstanceDetections) and isinstance(
            payload.mask, torch.Tensor
        ):
            tensors.append(payload.mask)
    elif isinstance(payload, ClassificationPrediction):
        tensors = [payload.class_id, payload.confidence]
    elif isinstance(payload, MultiLabelClassificationPrediction):
        tensors = [payload.class_ids, payload.confidence]
    elif isinstance(payload, tuple) and isinstance(payload[0], KeyPoints):
        points = payload[0]
        tensors = [
            points.xy,
            points.class_id,
            points.confidence,
            points.covariance,
            points.detection_confidence,
        ]
    else:
        tensors = [payload[0], payload[2], payload[3]]  # native detection iterator row
    summary = [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        for tensor in tensors
        if tensor is not None
    ]

    return summary


def _provenance(payload: Any) -> Any:
    if isinstance(payload, ClassificationPrediction):
        return payload.images_metadata
    if isinstance(payload, tuple):
        return (
            payload[0].image_metadata
            if isinstance(payload[0], KeyPoints)
            else payload[6]
        )
    return getattr(payload, "image_metadata", None)


def run_geometry(output_dir: Path) -> Dict[str, Any]:
    """Run the real nested workflow and check its numerical and grouping oracle.

    Args:
        output_dir: Directory receiving serialized rows, overlays and observations.

    Returns:
        Compact summary of the verified output geometry and sparse indices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [
        ImageData.from_tensor(pixels, image_id=name)
        for pixels, name in zip(
            make_root_pixels(), ["bright-root", "filtered-root", "empty-root"]
        )
    ]
    definition = json.loads((DEMO_DIR / "workflows/nested_geometry.json").read_text())
    plan = compile_workflow(definition, catalogue=create_demo_catalogue())
    result = plan.create_session().run({"images": images})

    own_raw = _raw(result, "own")[0][0][0]
    root_raw = _raw(result, "root")[0][0][0]
    selected_raw = _raw(result, "selected")[0][0][0]
    assert own_raw is root_raw is selected_raw
    assert own_raw.xyxy.device == images[0].device
    assert own_raw.xyxy.tolist() == [LOCAL_BOX]
    crops = _raw(result, "crops")
    assert [group.indices for group in crops] == [
        ((0, 0), (0, 2)),
        ((1, 0), (1, 2)),
        (),
    ]
    assert crops[2].parent_index == (2,)
    assert list(_raw(result, "count")) == [2, 2, 0]

    rows = result.rows()
    own, root = rows[0]["own"][0][0], rows[0]["root"][0][0]
    assert own is own_raw and root is not own
    assert own.xyxy.tolist() == [LOCAL_BOX]
    assert root.xyxy.tolist() == [EXPECTED_ROOT_BOX]
    assert root.image_metadata["parent_id"] == "bright-root"
    assert rows[0]["own"][2] == [None]
    assert rows[1]["own"] == [[None], [], [None]]
    assert rows[2]["own"] == []
    assert rows[2]["mosaic"].is_composite
    assert rows[2]["mosaic"].composite_sources == ()

    own_image = rows[0]["accepted_image"][0][0]
    _save_overlay(own_image, own, output_dir / "own-overlay.png")
    _save_overlay(images[0], root, output_dir / "root-overlay.png")
    _write_json(output_dir / "rows.json", result.rows(serialize=True))
    # A root sibling and serialized rows must not rewrite the raw/own carrier.
    assert own_raw.xyxy.tolist() == [LOCAL_BOX]
    assert result.rows()[0]["own"][0][0] is own_raw
    (own_entry,) = result.selections["own"].values()
    summary = {
        "filtered_prediction_paths": result.filtered_paths[own_entry],
        "local_box": own.xyxy.tolist()[0],
        "expected_root_box": EXPECTED_ROOT_BOX,
        "actual_root_box": root.xyxy.tolist()[0],
        "crop_indices": [list(group.indices) for group in crops],
        "filtered_sample": 1,
        "genuinely_empty_sample": 2,
        "own_and_root_share_raw_payload": own_raw is root_raw,
        "root_rows_are_independent": root is not own,
        "detection_ids": [row["detection_id"] for row in own.bboxes_metadata],
        "own_image_id": own_image.image_id,
        "immediate_parent": own_image.parent.to_dict(),
        "root_transform": own_image.root.to_dict(),
        "native_tensors": _tensor_summary(own),
        "own_provenance": own.image_metadata,
        "root_provenance": root.image_metadata,
    }
    _write_json(output_dir / "summary.json", summary)
    click.echo(f"  geometry: {LOCAL_BOX} -> {root.xyxy.tolist()[0]}")
    click.echo(
        "  groups: sparse crop positions [0, 2]; sample 1 filtered; sample 2 empty"
    )

    return summary


def run_gallery(output_dir: Path) -> Dict[str, Any]:
    """Round-trip every native family through compiled ingress and output rows.

    Args:
        output_dir: Directory receiving one faithful wire file per fixture.

    Returns:
        Family coverage and dtype/shape/provenance observations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image = ImageData.from_tensor(
        torch.zeros((3, 8, 10), dtype=torch.uint8), image_id="gallery"
    )
    fixtures = make_native_fixtures(image.prediction_metadata())
    records = []
    for name, kind, payload in fixtures:
        plan = compile_workflow(_passthrough(kind), catalogue=create_catalogue())
        session = plan.create_session()
        result = session.run({"value": payload})
        assert _raw(result, "value") is payload
        wire = json.loads(json.dumps(result.rows(serialize=True)[0]["value"]))
        restored_result = session.run({"value": wire})
        restored = _raw(restored_result, "value")
        assert _tensor_summary(restored) == _tensor_summary(payload)
        assert _provenance(restored) == _provenance(payload)
        assert restored_result.rows(serialize=True)[0]["value"] == wire
        _write_json(output_dir / f"{name}.json", wire)
        records.append(
            {
                "fixture": name,
                "kind": kind,
                "carrier": type(payload).__name__,
                "tensors": _tensor_summary(restored),
                "provenance_preserved": True,
                "wire_roundtrip_equal": True,
                "native_ingress_identity": True,
            }
        )
        click.echo(f"  gallery: {name} ({kind}) round-trip passed")
    assert {kind for _, kind, _ in fixtures} == {kind.name for kind in NATIVE_KINDS}
    summary = {"native_kind_count": len(NATIVE_KINDS), "fixtures": records}
    _write_json(output_dir / "summary.json", summary)

    return summary


def run_invalid(output_dir: Path) -> Dict[str, str]:
    """Show indexed ingress failures and the composite root-restoration boundary.

    Args:
        output_dir: Directory receiving actionable error messages.

    Returns:
        Error text for each deliberately invalid example.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bad_rows = Detections(
        xyxy=torch.zeros((2, 4)),
        class_id=torch.zeros(1, dtype=torch.int64),
        confidence=torch.ones(2),
    )
    valid = Detections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.zeros(1, dtype=torch.int64),
        confidence=torch.ones(1),
    )
    cases = [
        ("hwc-image", "image", torch.zeros((8, 10, 3), dtype=torch.uint8), "own"),
        ("float-image", "image", torch.zeros((3, 8, 10)), "own"),
        ("unequal-rows", "object_detection_prediction", bad_rows, "own"),
        ("unknown-coordinates", "object_detection_prediction", valid, "moon"),
    ]
    errors = {}
    for name, kind, payload, coordinates in cases:
        plan = compile_workflow(
            _passthrough(kind, coordinates=coordinates), catalogue=create_catalogue()
        )
        try:
            plan.create_session().run({"value": payload}).rows()
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
        else:
            raise AssertionError(f"Invalid case {name} unexpectedly succeeded")

    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}],
        "steps": [
            {
                "type": "v2/mosaic",
                "name": "mosaic",
                "images": "$inputs.images",
                "tile_size": 20,
            },
            {
                "type": "tensor_demo/synthetic_detections",
                "name": "predict",
                "image": "$steps.mosaic.image",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.predict.predictions",
                "coordinates_system": "root",
            }
        ],
    }
    plan = compile_workflow(definition, catalogue=create_demo_catalogue())
    pixels = [torch.full((3, 20, 20), 180, dtype=torch.uint8) for _ in range(2)]
    for name, images in [("multi-source-mosaic", pixels), ("empty-mosaic", [])]:
        result = plan.create_session().run({"images": images})
        assert _raw(result, "predictions").image_metadata["is_composite"]
        try:
            result.rows()
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
            assert "composite" in str(error).lower()
        else:
            raise AssertionError(
                f"Ambiguous root restoration of {name} unexpectedly succeeded"
            )
    definition["outputs"][0]["coordinates_system"] = "own"
    own_plan = compile_workflow(definition, catalogue=create_demo_catalogue())
    own_result = own_plan.create_session().run({"images": pixels})
    assert own_result.rows()[0]["predictions"] is _raw(own_result, "predictions")
    _write_json(output_dir / "mosaic-own.json", own_result.rows(serialize=True))
    for name, error in errors.items():
        click.echo(f"  expected error [{name}]: {error}")
    _write_json(output_dir / "errors.json", errors)

    return errors


def _write_index(output_dir: Path, results: Dict[str, Any]) -> None:
    sections = [
        "<h1>Tensor-native Workflows V2</h1>"
        "<p>Deterministic synthetic fixtures; no model inference.</p>"
    ]
    if "geometry" in results:
        sections.append("""
            <h2>Nested geometry</h2>
            <p>Local [2,4,6,8] → root [44,98,52,114].</p>
            <figure><img src="geometry/root-overlay.png" width="400">
              <figcaption>300×400 workflow root, restored box</figcaption></figure>
            <figure><img src="geometry/own-overlay.png" width="200"
              style="image-rendering:pixelated">
              <figcaption>20×20 nested crop, local box (enlarged)</figcaption></figure>
            <p><a href="geometry/summary.json">IDs, transforms, dtype, device and indices</a>
              · <a href="geometry/rows.json">Serialized output rows</a></p>
        """)
    if "gallery" in results:
        sections.append("""
            <h2>Native family gallery</h2>
            <table><tr><th>Fixture / wire</th><th>Kind</th>
              <th>Tensor shapes and dtypes</th></tr>
        """)
        for fixture in results["gallery"]["fixtures"]:
            name = fixture["fixture"]
            tensors = "; ".join(
                f"{item['shape']} {item['dtype']} on {item['device']}"
                for item in fixture["tensors"]
            )
            sections.append(
                f'<tr><td><a href="gallery/{name}.json">{name}</a></td>'
                f'<td>{fixture["kind"]}</td><td>{html.escape(tensors)}</td></tr>'
            )
        sections.append("""
            </table><p>All native ingress identities and wire round-trips passed;
            shape, dtype and provenance retained. JSON decoding allocates on
            the receiving CPU.</p>
        """)
    if "invalid" in results:
        sections.append(
            "<h2>Expected boundary errors</h2><pre>"
            + html.escape(json.dumps(results["invalid"], indent=2))
            + "</pre>"
        )
    document = """<!doctype html>
        <html lang="en"><meta charset="utf-8">
        <title>Tensor-native V2 demo</title>
        <style>
          body {font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px}
          td,th {text-align:left;border-bottom:1px solid #ddd;padding:8px}
          pre {white-space:pre-wrap}
          figure {display:inline-block;vertical-align:top;margin:10px}
          figcaption {font-size:14px;color:#555}
        </style><body>
        """ + "\n".join(sections) + "</body></html>"
    (output_dir / "index.html").write_text(document)


@click.command()
@click.option(
    "--scenario",
    type=click.Choice(["geometry", "gallery", "invalid", "all"]),
    default="all",
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory receiving every generated artifact.",
)
def main(scenario: str, output_dir: Path) -> None:
    """Execute the selected native-media examples and write a browsable index.

    Args:
        scenario: Example group, or all groups.
        output_dir: Destination for generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runners = {"geometry": run_geometry, "gallery": run_gallery, "invalid": run_invalid}
    names = list(runners) if scenario == "all" else [scenario]
    results = {name: runners[name](output_dir / name) for name in names}
    _write_json(output_dir / "summary.json", results)
    _write_index(output_dir, results)
    click.echo(f"All selected scenarios passed. Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
