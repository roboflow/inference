"""Batched detection painters and their optional shared drawing preparation.

Every block takes whole batches (``batch="always"``), so the engine calls it
once per batch. Painters queue the batch's work on one CUDA stream and wait
once before returning (``streams.ReadyScope``). Images never leave the device.

Routes for boxes plus labels on ``B`` images::

    route                                            image   host       painter
                                                     copies  transfers  waits
    detection_visualization (fused, recommended)       B        1         1
    detection_drawing -> box -> label_in_place         B        1         2
    detection_drawing -> box -> label                  2B       1         2
    box -> label_in_place, both bound to Detections    B        2         2

A host transfer is one device-to-host copy of the batch's boxes, class ids
and confidences (``drawing.prepare_drawings``); the host waits for it. A
painter bound to native ``Detections`` makes it itself; bound to the output
of ``roboflow_core/detection_drawing@v1`` it makes none, so two painters share one.
A painter wait is the one host wait at the end of each painter call.
Drawing also waits for each image's box-table upload on CUDA (``kernels``);
those uploads are not included in the table's painter-wait count.

Mutation is declared, never implied. Copy painters return new images and
leave their inputs untouched. ``roboflow_core/simple_label_visualization_in_place@v1`` changes
its input image (``mutates``); bind it only to an image an earlier step of
this workflow created, such as the box painter's output. The compiler reports
conflicting readers of that image; it does not check who owns a passive
workflow input, so binding a caller's frame overwrites the caller's pixels.

Pixels are drawn by ``kernels.TensorKernels`` on CPU and CUDA. Workflow
definitions do not choose how pixels are drawn: a host that has other
implementations of these blocks adds them to its catalogue
(``Catalogue.with_contributions``).

Output images keep the input's identity, provenance, video metadata and
geometry (``ImageData.with_pixels``). Copies are contiguous and allocated on
the painter's stream; a consumer that queues asynchronous work on another
stream must ``record_stream`` them before dropping them.
"""


from roboflow_workflows.v2.blocks.visualization.native.detection_drawing.v1 import DetectionDrawingBlock
from roboflow_workflows.v2.blocks.visualization.native.detection_visualization.v1 import DetectionVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.native.simple_box_visualization.v1 import BoxVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.native.simple_label_visualization.v1 import LabelVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.native.simple_label_visualization_in_place.v1 import InPlaceLabelVisualizationBlock


BLOCKS = (
    DetectionDrawingBlock,
    BoxVisualizationBlock,
    LabelVisualizationBlock,
    DetectionVisualizationBlock,
    InPlaceLabelVisualizationBlock,
)
