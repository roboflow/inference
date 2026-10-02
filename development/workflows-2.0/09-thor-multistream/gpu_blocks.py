"""The 08 live detection blocks, made correct for CUDA images.

    image (CUDA CHW RGB) ─┬─ detector (08) ── predictions ─┐
                          └── boxes (clone + paint) ── labels (in place)

V2 hands a block result on as soon as the block returns. Its readiness
contract knows ``concurrent.futures.Future`` values only, nothing about CUDA.
So every block here returns only after its own GPU work has finished:

    block           where its GPU work runs           ready at return because
    detector        the TRT model's own streams        the model synchronizes
                                                       them inside each of
                                                       pre_process, forward
                                                       and post_process
    boxes, labels   one stream per block and thread    _ReadyOnReturn waits on
                                                       an event recorded after
                                                       this call's work

Waiting on the host blocks only the calling worker. In a V2 pipeline other
pulses keep running other stages meanwhile. Nothing here calls
``torch.cuda.synchronize()``.

Because every block has finished reading its inputs when it returns, the
caller may release the source frame as soon as the result is out, and a
tensor never outlives the stream work that uses it. No ``record_stream`` is
needed.

The painters reuse the 08 drawing code. The label painter also stages its
per-frame paste table through a pinned ring, as V1's tensor label block does.
"""

import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Tuple

import torch
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _LabelSprite,
    _PinnedSlabRing,
    gpu_paste_label_sprites,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND

LIVE_DETECTION_DIR = Path(__file__).resolve().parent.parent / "08-live-detection"
if str(LIVE_DETECTION_DIR) not in sys.path:
    sys.path.append(str(LIVE_DETECTION_DIR))

from detection_blocks import (  # noqa: E402
    PALETTE,
    BoxVisualization,
    LabelVisualization,
    ObjectDetector,
    _class_and_confidence_labels,
    _elapsed_ms,
    _require_rgb,
)

from inference_models.models.base.object_detection import Detections  # noqa: E402


class _ReadyOnReturn:
    """Run a block's CUDA work on its own stream and wait for it before return.

    Each worker thread gets its own stream per block, so two pulses in
    different stages never queue behind each other on one stream. On a CPU
    image this does nothing.
    """

    def __init__(self):
        self._local = threading.local()

    @contextmanager
    def work(self, device: torch.device) -> Iterator[None]:
        if device.type != "cuda":
            yield
            return

        stream = self._stream(device)
        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            # Also on failure: queued kernels may still read the input frame.
            done = torch.cuda.Event()
            done.record(stream)
            done.synchronize()

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        streams: Dict[torch.device, torch.cuda.Stream] = (
            self._local.__dict__.setdefault("streams", {})
        )
        if device not in streams:
            streams[device] = torch.cuda.Stream(device=device)

        return streams[device]


class GpuBoxVisualization(BoxVisualization):
    """The 08 box painter; returns a CUDA image whose pixels are written.

    ``draw_ms`` covers the copy, the paint and the wait for both.
    """

    type = "thor_detection/box_visualization@v1"

    def __init__(self):
        self._cuda = _ReadyOnReturn()

    def run(self, *, image: ImageData, predictions: Detections) -> Dict[str, Any]:
        """Paint the box borders of ``predictions`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified. Its pixels must be complete.
            predictions: Detections to draw; tensors must be complete.

        Returns:
            ``image``: the painted copy, ready to read; ``draw_ms``: wall time
            including the GPU work.

        Raises:
            ValueError: When the image is not RGB.
        """
        started = perf_counter()
        with self._cuda.work(image.tensor_image.device):
            result = super().run(image=image, predictions=predictions)
        result["draw_ms"] = _elapsed_ms(started)

        return result


class GpuLabelVisualization(LabelVisualization):
    """The 08 label painter; the image is fully labelled when ``run`` returns.

    Same geometry, sprites and paste as 08. The one change besides the wait:
    the paste table goes through a pinned ring (``_PinnedSlabRing``), as in
    V1's tensor label block, instead of a new pinned buffer per frame.
    """

    type = "thor_detection/label_visualization@v1"

    def __init__(self):
        super().__init__()
        self._cuda = _ReadyOnReturn()
        self._table_ring = _PinnedSlabRing()

    def run(self, *, image: ImageData, predictions: Detections) -> Dict[str, Any]:
        """Paste one label per detection above its box's top-left corner.

        Args:
            image: Contiguous RGB image; modified in place.
            predictions: Detections whose ``image_metadata["class_names"]``
                names every class id.

        Returns:
            ``image``: the same image object, ready to read; ``draw_ms``: wall
            time including the GPU work.

        Raises:
            ValueError: On a non-RGB or non-contiguous image, or a class id
                without a name.
        """
        _require_rgb(image, block=self.type)
        if not image.tensor_image.is_contiguous():
            raise ValueError(
                f"{self.type} paints in place and needs a contiguous image tensor; "
                "bind the box painter's output or another owned contiguous image"
            )

        started = perf_counter()
        scene = image.tensor_image
        with self._cuda.work(scene.device):
            sprites, origins = self._sprites_and_origins(predictions, scene=scene)
            gpu_paste_label_sprites(scene, sprites, origins, self._table_ring)

        result = {"image": image, "draw_ms": _elapsed_ms(started)}

        return result

    def _sprites_and_origins(
        self, predictions: Detections, *, scene: torch.Tensor
    ) -> Tuple[List[_LabelSprite], List[Tuple[int, int]]]:
        # The label loop of the 08 LabelVisualization.run, unchanged.
        labels = _class_and_confidence_labels(predictions)
        frame_hw = (int(scene.shape[1]), int(scene.shape[2]))
        corners = predictions.xyxy[:, :2].cpu().numpy().astype(int)
        class_ids = predictions.class_id.cpu().numpy()
        sprites: List[_LabelSprite] = []
        origins: List[Tuple[int, int]] = []
        for label, (x, y), class_id in zip(labels, corners, class_ids):
            placed = self._place(
                label,
                anchor=(int(x), int(y)),
                background_bgr=PALETTE.by_idx(int(class_id)).as_bgr(),
                frame_hw=frame_hw,
                device=scene.device,
            )
            if placed is not None:
                sprites.append(placed[0])
                origins.append(placed[1])

        return sprites, origins


BLOCKS = (ObjectDetector, GpuBoxVisualization, GpuLabelVisualization)


def create_catalogue() -> Catalogue:
    """Collect ``BLOCKS``: the 08 detector and the two CUDA-ready painters.

    Returns:
        A catalogue with ``live_detection/object_detector@v1``,
        ``thor_detection/box_visualization@v1`` and
        ``thor_detection/label_visualization@v1``.
    """
    catalogue = Catalogue(
        BLOCKS,
        kinds=[IMAGE_KIND, OBJECT_DETECTION_PREDICTION_KIND, FLOAT_KIND],
    )

    return catalogue
