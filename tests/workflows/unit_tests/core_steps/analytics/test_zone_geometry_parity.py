import importlib.util
from pathlib import Path

import numpy as np
import supervision as sv

_MODULE_PATH = (
    Path(__file__).parents[5]
    / "inference/core/workflows/core_steps/analytics/_zone_geometry.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location("_zone_geometry", _MODULE_PATH)
_ZONE_GEOMETRY = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_ZONE_GEOMETRY)
LeanLineZone = _ZONE_GEOMETRY.LeanLineZone
LeanPolygonZone = _ZONE_GEOMETRY.LeanPolygonZone
anchor_coordinates = _ZONE_GEOMETRY.anchor_coordinates


def test_anchor_coordinates_matches_supervision_for_every_position() -> None:
    rng = np.random.default_rng(731)
    top_left = rng.uniform(-100, 500, size=(50, 2))
    xyxy = np.concatenate(
        [top_left, top_left + rng.uniform(0, 100, size=(50, 2))], axis=1
    ).astype(float)
    detections = sv.Detections(xyxy=xyxy)

    for position in sv.Position:
        if position == sv.Position.CENTER_OF_MASS:
            try:
                anchor_coordinates(xyxy, position)
            except ValueError as error:
                lean_error = error
            else:
                raise AssertionError("Lean anchor calculation did not raise")
            try:
                detections.get_anchors_coordinates(position)
            except ValueError as error:
                supervision_error = error
            else:
                raise AssertionError("Supervision anchor calculation did not raise")
            assert str(lean_error) == str(supervision_error)
        else:
            np.testing.assert_array_equal(
                anchor_coordinates(xyxy, position),
                detections.get_anchors_coordinates(position),
            )


def test_lean_line_zone_matches_supervision() -> None:
    rng = np.random.default_rng(904)
    scenarios_per_anchor = 70

    for triggering_anchors in (
        None,
        [sv.Position.CENTER],
        [sv.Position.BOTTOM_CENTER],
    ):
        for scenario in range(scenarios_per_anchor):
            line_kind = scenario % 3
            if line_kind == 0:
                start = (0.0, float(rng.uniform(20, 180)))
                end = (200.0, start[1])
            elif line_kind == 1:
                start = (float(rng.uniform(20, 180)), 0.0)
                end = (start[0], 200.0)
            else:
                start = tuple(rng.uniform(-50, 50, size=2))
                delta = rng.uniform(-160, 160, size=2)
                if np.linalg.norm(delta) < 1:
                    delta[0] += 40
                end = tuple(np.asarray(start) + delta)

            lean_zone = LeanLineZone(start, end, triggering_anchors)
            supervision_kwargs = (
                {}
                if triggering_anchors is None
                else {"triggering_anchors": triggering_anchors}
            )
            supervision_zone = sv.LineZone(
                start=sv.Point(*start), end=sv.Point(*end), **supervision_kwargs
            )

            n = int(rng.integers(0, 31))
            tracker_ids = np.arange(n, dtype=int) + scenario * 100
            centers = rng.uniform(-100, 300, size=(n, 2))
            sizes = rng.uniform(0, 50, size=(n, 2))
            velocity = rng.normal(0, 8, size=(n, 2))
            if n:
                centers[0] = np.asarray(start)
                sizes[0] = 30

            for frame in range(40):
                centers += velocity + rng.normal(0, 2, size=(n, 2))
                if n and frame % 8 == 0:
                    velocity[: max(1, n // 5)] *= -1
                xyxy = np.concatenate(
                    [centers - sizes / 2, centers + sizes / 2], axis=1
                ).astype(float)

                lean_in, lean_out = lean_zone.trigger(xyxy, tracker_ids)
                supervision_in, supervision_out = supervision_zone.trigger(
                    sv.Detections(xyxy=xyxy, tracker_id=tracker_ids)
                )

                np.testing.assert_array_equal(lean_in, supervision_in)
                np.testing.assert_array_equal(lean_out, supervision_out)
                assert lean_zone.in_count == supervision_zone.in_count
                assert lean_zone.out_count == supervision_zone.out_count


def test_lean_polygon_zone_matches_supervision() -> None:
    rng = np.random.default_rng(117)

    for triggering_anchors in (
        (sv.Position.CENTER,),
        (sv.Position.BOTTOM_CENTER,),
    ):
        for scenario in range(220):
            vertex_count = int(rng.integers(3, 9))
            center = rng.uniform(80, 220, size=2)
            angles = np.sort(rng.uniform(0, 2 * np.pi, size=vertex_count))
            radii = rng.uniform(15, 75, size=vertex_count)
            polygon = np.stack(
                [
                    center[0] + np.cos(angles) * radii,
                    center[1] + np.sin(angles) * radii,
                ],
                axis=1,
            )

            n = int(rng.integers(0, 31))
            top_left = rng.uniform(-80, 360, size=(n, 2))
            sizes = rng.uniform(0, 90, size=(n, 2))
            if n:
                sizes[0] = 0
            xyxy = np.concatenate([top_left, top_left + sizes], axis=1).astype(float)

            lean_zone = LeanPolygonZone(polygon, triggering_anchors)
            supervision_zone = sv.PolygonZone(polygon, triggering_anchors)
            np.testing.assert_array_equal(
                lean_zone.trigger(xyxy),
                supervision_zone.trigger(sv.Detections(xyxy=xyxy)),
            )
            assert lean_zone.current_count == supervision_zone.current_count
