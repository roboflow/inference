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

## Standalone stream manager

The stream manager (`python -m streamvision`) needs `streamvision[webrtc]`; the
library parts (`streamvision.camera`, `streamvision.stream`, the TCP client and
entities) work without it.

```bash
STREAM_MANAGER_PORT=7070 python -m streamvision \
    --host-factory my_package.host:create_host \
    --host-setting api_key=... --warm-pipelines 1
```

`--host-factory` names a trusted callable returning a pipeline host; its module
is imported before the runtime. `scripts/streamvision_isolation_probe.py`
verifies an installed wheel runs without `inference`.
