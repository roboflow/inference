# streamvision

`streamvision` holds the host-neutral pieces of Roboflow Inference's video
stack: camera acquisition (`VideoSource` and its producers), the host-neutral
`InferencePipeline`, the stream-manager TCP client and its wire entities, and
the stream-manager runtime that hosts pipelines as subprocesses. It is a
sibling distribution to `roboflow-workflows`, installable on its own or
embedded in the full Inference server.

The historical `inference.core.interfaces.{camera,stream,stream_manager}`
import paths resolve to the same modules here when `inference` is installed.

## Hardware decoding

Hardware decoders are optional. Without them `VideoSource` decodes on the CPU with
OpenCV. They are tried only when `enable_tensor_data_representation` is set in the
installed `StreamsConfiguration`. A decoder that cannot be used is skipped with a
warning that names the reason.

| Decoder | Sources | Needs |
|---|---|---|
| `PyNvVideoCodecFrameProducer` | video files, NVIDIA GPU | `pip install "streamvision[nvdec]"` (Linux x86_64 and Windows x64) and a CUDA build of `torch` |
| `GstreamerCudaVideoFrameProducer` | streams and files, NVIDIA GPU | GStreamer with the `nvcodec` plugin and the Roboflow CUDA tensor bridge library; both ship in the Roboflow GPU Docker images |
| `JetsonVideoFrameProducer` | every source, NVIDIA Jetson | GStreamer with the Jetson elements and the Roboflow Jetson tensor bridge library; both ship in the Roboflow Jetson Docker images |

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
