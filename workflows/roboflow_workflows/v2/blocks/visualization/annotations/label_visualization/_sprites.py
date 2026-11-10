"""Bounded, scene-independent label sprites with exact reference fallback.

Canonical layout supplies frame-coordinate labels. Single nonempty lines use
the V1 sprite renderer, including its test for scene-dependent antialiasing.
Multiline text, unsupported devices and oversized patches use the reference
renderer. All sprites are prepared before any pixels are written.
"""

import threading
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Sequence

import torch
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _measure_label,
    _PinnedSlabRing,
    _render_label_sprite,
    _SceneDependentLabelError,
    gpu_paste_label_sprites,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    Label,
    paint_labels,
)

_MAX_SPRITES = 256
_MAX_CACHE_BYTES = 16 * 1024 * 1024
_MAX_PATCH_PIXELS = 262144
_measurement = lru_cache(maxsize=4096)(_measure_label)


class _ReadyScope:
    """Join caller work, then finish this thread's CUDA painting before return."""

    def __init__(self) -> None:
        self._local = threading.local()

    @contextmanager
    def work(self, device: torch.device) -> Iterator[None]:
        if device.type != "cuda":
            yield
            return

        streams = self._local.__dict__.setdefault("streams", {})
        if device not in streams:
            streams[device] = torch.cuda.Stream(device=device)
        stream = streams[device]
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            done = torch.cuda.Event()
            done.record(stream)
            done.synchronize()


class _LabelSprites:
    """Per-implementation cache; the engine invokes an instance serially."""

    def __init__(self) -> None:
        self._sprites = OrderedDict()
        self._cache_bytes = 0
        self._retired = []
        self._scope = _ReadyScope()
        self._table_ring = _PinnedSlabRing()

    def paint(self, scene: torch.Tensor, labels: Sequence[Label]) -> None:
        """Paint cached opaque sprites, or use the reference for the whole call.

        Args:
            scene: Contiguous RGB canvas, modified in place.
            labels: Canonical labels in drawing order.
        """
        if scene.device.type not in ("cpu", "cuda"):
            paint_labels(scene, labels)
            return

        try:
            with self._scope.work(scene.device):
                placements = self._prepare(scene, labels)
                if placements is None:
                    paint_labels(scene, labels)
                else:
                    sprites = [item[0] for item in placements]
                    origins = [item[1] for item in placements]
                    gpu_paste_label_sprites(
                        scene, sprites, origins, table_ring=self._table_ring
                    )
        finally:
            # Eviction must not release pinned uploads before the final wait,
            # including when a later label sends the whole call to fallback.
            self._retired.clear()

    def _prepare(self, scene: torch.Tensor, labels: Sequence[Label]):
        height, width = int(scene.shape[1]), int(scene.shape[2])
        placements = []
        for label in labels:
            if len(label.lines) != 1:
                return None

            text, text_x, text_y = label.lines[0]
            style = label.style
            measured = _measurement(text, style.scale, style.thickness, style.padding)
            bx1, by1, bx2, by2 = label.background
            # The renderer's single-line origin must equal the canonical one.
            if (text_x, text_y) != (
                bx1 + style.padding,
                by1 + measured.height_padded - style.padding,
            ):
                return None

            margin = measured.margin
            left, top = bx1 - margin, by1 - margin
            right, bottom = bx2 + 1 + margin, by2 + 1 + margin
            x0, y0 = max(left, 0), max(top, 0)
            x1, y1 = min(right, width), min(bottom, height)
            if x1 <= x0 or y1 <= y0:
                continue
            if (x1 - x0) * (y1 - y0) > _MAX_PATCH_PIXELS:
                return None

            box = (bx1 - x0, by1 - y0, bx2 - x0, by2 - y0)
            shape = (y1 - y0, x1 - x0)
            edges = (left < 0, top < 0, right > width, bottom > height)
            key = (
                text,
                style,
                label.text_rgb,
                label.background_rgb,
                str(scene.device),
                width,
                box,
                shape,
                edges,
            )
            if key in self._sprites:
                sprite, _ = self._sprites[key]
                self._sprites.move_to_end(key)
            else:
                try:
                    sprite = _render_label_sprite(
                        measurement=measured,
                        text_color_bgr=label.text_rgb[::-1],
                        background_color_bgr=label.background_rgb[::-1],
                        text_scale=style.scale,
                        text_thickness=style.thickness,
                        text_padding=style.padding,
                        border_radius=style.border_radius,
                        device=scene.device,
                        box_in_canvas=box,
                        canvas_hw=shape,
                        frame_edge_sides=edges,
                    )
                except _SceneDependentLabelError:
                    sprite = None
                # Conservative accounting includes host coordinates, device
                # colors, flat indices and staging. Width is part of the key,
                # so each sprite's flat-index cache has only one entry.
                cost = 0 if sprite is None else sprite.pixel_count * 32
                self._sprites[key] = (sprite, cost)
                self._cache_bytes += cost
                while (
                    len(self._sprites) > _MAX_SPRITES
                    or self._cache_bytes > _MAX_CACHE_BYTES
                ):
                    retired, retired_cost = self._sprites.popitem(last=False)[1]
                    self._cache_bytes -= retired_cost
                    self._retired.append(retired)

            if sprite is None:
                return None
            placements.append((sprite, (x0, y0)))

        return placements
