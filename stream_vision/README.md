# streamvision

`streamvision` holds the host-neutral pieces of Roboflow Inference's video
stack: camera acquisition (`VideoSource` and its producers), the host-neutral
`InferencePipeline`, the stream-manager TCP client and its wire entities, and
the stream-manager runtime that hosts pipelines as subprocesses. It is a
sibling distribution to `roboflow-workflows`, installable on its own or
embedded in the full Inference server.

The historical `inference.core.interfaces.{camera,stream,stream_manager}`
import paths resolve to the same modules here when `inference` is installed.
No `gpu`/`jetson` extra is declared: GStreamer/NVDEC support is provisioned by
the Docker images, not pinned as a PyPI dependency in `requirements/`.
