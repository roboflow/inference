"""Native OBB Workflow transport. Ordered corners carry the front through warps.

Inside a Workflow the first edge points along the native positive width/front
axis. Perspective/anisotropic warps retain that order as quadrilaterals; they
are never silently refitted into native rectangle annotations.
"""

import math

import cv2
import numpy as np
import supervision as sv

NATIVE_OBB = "native_obb"
CORNERS = "xyxyxyxy"


def _corners(obb):
    cx, cy, w, h, angle = obb
    c, s = math.cos(angle), math.sin(angle)
    local = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def _rectangle(corners):
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Malformed native OBB ordered corners")
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if not (np.all(turns > 0) or np.all(turns < 0)):
        raise ValueError(
            "Native OBB corners must form a nondegenerate convex footprint"
        )
    u, v = corners[1] - corners[0], corners[3] - corners[0]
    w, h = np.linalg.norm(u), np.linalg.norm(v)
    if w <= 0 or h <= 0:
        raise ValueError("Native OBB cannot have a zero side")
    if (
        not np.allclose(
            corners[0] + corners[2], corners[1] + corners[3], atol=1e-5, rtol=1e-6
        )
        or abs(np.dot(u, v)) > 1e-6 * w * h
    ):
        return None
    center = corners.mean(0)
    return [
        float(center[0]),
        float(center[1]),
        float(w),
        float(h),
        math.atan2(u[1], u[0]),
    ]


def native_obb_from_inference(response):
    """Parse native tuples or ordered Workflow quadrilaterals without an AABB cast.

    Args:
        response: Native inference response dict, including image dimensions.

    Returns:
        Supervision transport with float ordered corners and class/front metadata.

    Raises:
        ValueError: If geometry, direction, class semantics or identity are absent.
    """
    if response.get("task_type") != "oriented-object-detection":
        raise ValueError("Native OBB task identity is required")
    rows = response["predictions"]
    names, ids, confidences, corners, headings, detection_ids = [], [], [], [], [], []
    class_heading = response["class_heading"]
    if not isinstance(class_heading, dict) or any(
        type(value) is not bool for value in class_heading.values()
    ):
        raise ValueError("Native OBB requires frozen boolean class semantics")
    for row in rows:
        name = row["class"]
        if (
            type(row.get("heading")) is not bool
            or class_heading.get(name) is not row["heading"]
        ):
            raise ValueError("Native OBB heading must match frozen class semantics")
        if "obb" in row:
            obb = np.asarray(row["obb"], dtype=float)
            if obb.shape != (5,) or not np.isfinite(obb).all() or (obb[2:4] <= 0).any():
                raise ValueError("Malformed native OBB geometry")
            polygon = _corners(obb)
        elif row.get("geometry_type") == "quadrilateral":
            polygon = np.asarray(row["corners"], dtype=float)
            _rectangle(polygon)
            if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
                raise ValueError("Malformed native OBB quadrilateral")
        else:
            raise ValueError("Native geometry is required")
        if not isinstance(row.get("detection_id"), str) or not row["detection_id"]:
            raise ValueError("Native OBB detection identity is required")
        names.append(name)
        ids.append(row["class_id"])
        confidences.append(row["confidence"])
        corners.append(polygon)
        headings.append(row["heading"])
        detection_ids.append(row["detection_id"])
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
    if not np.isfinite(confidences).all() or any(
        not 0 <= value <= 1 for value in confidences
    ):
        raise ValueError("Native OBB confidence must be finite and in [0,1]")
    bounds = (
        np.concatenate((corners.min(1), corners.max(1)), axis=1)
        if rows
        else np.empty((0, 4))
    )
    height, width = response["image"]["height"], response["image"]["width"]
    result = sv.Detections(
        xyxy=bounds,
        class_id=np.asarray(ids, dtype=int),
        confidence=np.asarray(confidences, dtype=float),
        data={
            CORNERS: corners,
            "heading": np.asarray(headings, dtype=bool),
            "class_name": np.asarray(names, dtype=str),
            "detection_id": np.asarray(detection_ids, dtype=str),
            NATIVE_OBB: np.ones(len(rows), dtype=bool),
            "image_dimensions": np.tile([height, width], (len(rows), 1)),
        },
        metadata={
            NATIVE_OBB: True,
            "class_heading": dict(class_heading),
            "image_dimensions": [height, width],
        },
    )
    return result


def serialise_native_obb(detections):
    """Serialize a native Workflow result from its current ordered corners.

    Args:
        detections: Native supervision transport after transformations/selection.

    Returns:
        Native response; nonrectangular warps explicitly emit quadrilaterals.

    Raises:
        ValueError: If front metadata or geometry has been dropped.
    """
    detections = _native_obb_to_sv(detections)
    rows = []
    if not isinstance(detections.metadata.get("class_heading"), dict):
        raise ValueError("Native OBB frozen class_heading metadata is required")
    class_heading = dict(detections.metadata["class_heading"])
    for i in range(len(detections)):
        data = detections.data
        if CORNERS not in data or "heading" not in data:
            raise ValueError("Native OBB corners and heading are required")
        corners = np.asarray(data[CORNERS][i], dtype=np.float64)
        rectangle = _rectangle(corners)
        name = str(data["class_name"][i])
        if not isinstance(data["heading"][i], (bool, np.bool_)):
            raise ValueError("Native OBB heading must be boolean")
        heading = bool(data["heading"][i])
        if name in class_heading and class_heading[name] != heading:
            raise ValueError("Conflicting native class heading semantics")
        class_heading[name] = heading
        row = {
            "class": name,
            "class_id": int(detections.class_id[i]),
            "confidence": float(detections.confidence[i]),
            "heading": heading,
            "detection_id": str(data["detection_id"][i]),
        }
        if rectangle is None:
            row.update(geometry_type="quadrilateral", corners=corners.tolist())
        else:
            row["obb"] = rectangle
        if "parent_id" in data:
            row["parent_id"] = str(data["parent_id"][i])
        rows.append(row)
    dimensions = detections.metadata.get("image_dimensions")
    if len(detections) and "image_dimensions" in detections.data:
        dimensions = detections.data["image_dimensions"][0]
    if dimensions is None:
        raise ValueError("Native OBB image dimensions are required")
    result = {
        "predictions": rows,
        "image": {"height": int(dimensions[0]), "width": int(dimensions[1])},
        "task_type": "oriented-object-detection",
        "class_heading": class_heading,
    }
    return result


def deserialize_native_obb(parameter, detections):
    """Deserialize a native Workflow input.

    Args:
        parameter: Workflow parameter name.
        detections: Native response dict or supervision transport.

    Returns:
        Validated native supervision transport.
    """
    from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION

    result = (
        native_obb_from_inference(detections)
        if isinstance(detections, dict)
        else _native_obb_to_sv(detections)
    )
    serialise_native_obb(result)
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        result = native_obb_to_tensor(result)
    return result


def annotate_native_obb(image, detections, *, thickness=2):
    """Draw each current footprint and the designated front edge.

    Args:
        image: BGR uint8 HWC image, modified in place.
        detections: Native transport.
        thickness: Line width in pixels.

    Returns:
        Annotated input image.
    """
    for i, corners in enumerate(detections.data[CORNERS]):
        points = np.rint(corners).astype(np.int32)
        cv2.polylines(image, [points], True, (251, 81, 163), thickness)
        if detections.data["heading"][i]:
            cv2.line(
                image, tuple(points[1]), tuple(points[2]), (51, 182, 255), thickness + 1
            )
    return image


def native_obb_to_tensor(detections):
    """Convert native corner transport to the standard tensor detection carrier.

    Args:
        detections: Native supervision detections with image/crop lineage.

    Returns:
        Tensor detections; ordered corners and heading remain per-row metadata.
    """
    import torch

    from inference_models.models.base.object_detection import Detections

    metadata = dict(detections.metadata)
    if len(detections):
        for key in detections.data:
            if key not in {
                CORNERS,
                "heading",
                "class_name",
                "detection_id",
                NATIVE_OBB,
            }:
                value = detections.data[key][0]
                metadata[key] = (
                    value.tolist() if isinstance(value, np.ndarray) else value
                )
    metadata["class_names"] = {
        int(c): str(n)
        for c, n in zip(detections.class_id, detections.data["class_name"])
    }
    entries = [
        {key: value[i] for key, value in detections.data.items()}
        for i in range(len(detections))
    ]
    result = Detections(
        xyxy=torch.as_tensor(detections.xyxy),
        class_id=torch.as_tensor(detections.class_id),
        confidence=torch.as_tensor(detections.confidence),
        image_metadata=metadata,
        bboxes_metadata=entries,
    )
    return result


def _native_obb_to_sv(detections):
    if isinstance(detections, sv.Detections):
        return detections
    entries = detections.bboxes_metadata or []
    keys = set().union(*(e.keys() for e in entries))
    data = {key: np.asarray([e[key] for e in entries]) for key in keys}
    metadata = dict(detections.image_metadata or {})
    # Tensor coordinate operations update image lineage at image level; row-level
    # copies may still describe the old crop, so always prefer the current frame.
    for key in (
        "image_dimensions",
        "parent_id",
        "parent_coordinates",
        "parent_dimensions",
        "root_parent_coordinates",
        "root_parent_dimensions",
        "root_parent_id",
    ):
        if key in metadata:
            data[key] = np.asarray([metadata[key]] * len(entries))
    result = sv.Detections(
        xyxy=detections.xyxy.detach().cpu().numpy(),
        confidence=detections.confidence.detach().cpu().numpy(),
        class_id=detections.class_id.detach().cpu().numpy(),
        data=data,
        metadata=metadata,
    )
    return result


def merge_native_semantics(first, second):
    """Reconcile frozen capabilities before combining native prediction rows.

    Args:
        first: NumPy or tensor predictions.
        second: Predictions in the same representation and image frame.

    Returns:
        Combined frozen class map for native inputs, otherwise None.

    Raises:
        ValueError: For mixed tasks or conflicting frozen class capabilities.
    """
    maps = []
    native = []
    for prediction in (first, second):
        metadata = (
            prediction.metadata
            if isinstance(prediction, sv.Detections)
            else (prediction.image_metadata or {})
        )
        native.append(bool(metadata.get(NATIVE_OBB)))
        maps.append(metadata.get("class_heading"))
    if not any(native):
        return None
    if not all(native) or any(not isinstance(value, dict) for value in maps):
        raise ValueError(
            "Native OBB combination requires frozen native semantics on both inputs"
        )
    combined = dict(maps[0])
    for name, heading in maps[1].items():
        if name in combined and combined[name] is not heading:
            raise ValueError("Conflicting native class heading semantics")
        combined[name] = heading
    return combined
