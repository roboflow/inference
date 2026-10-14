# streamvision

`streamvision` holds the host-neutral pieces of Roboflow Inference's video
stack: camera acquisition (`VideoSource` and its producers), the host-neutral
`InferencePipeline`, the stream-manager TCP client and its wire entities, and
the stream-manager runtime that hosts pipelines as subprocesses. It is a
sibling distribution to `roboflow-workflows`, installable on its own or
embedded in the full Inference server.

The historical `inference.core.interfaces.{camera,stream,stream_manager}`
import paths resolve to the same modules here when `inference` is installed.

## Configuration

Settings live in one `StreamsConfiguration` per process. Install it with
`streamvision.stream.configuration.configure_process` before importing any runtime
module: the values are frozen into module constants at the first import. Without an
installed configuration the defaults apply. Installing a different configuration
afterwards raises `StreamsConfigurationError`, which names the differing fields.

In a process that also uses `inference`, import it first:

```python
import inference.core  # or: from inference import InferencePipeline
import streamvision.stream.pipeline
```

`inference.core` installs the configuration built from the environment variables of
`inference`. A plain `import inference` is lazy and installs nothing. With the opposite
order and an environment that differs from the defaults, `import inference.core` raises
`StreamsConfigurationError` or `WorkflowEnvironmentConfigurationError`.

Processes started by the stream manager install the configuration passed by the launcher
before they import the host factory module.

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

The stream manager (`python -m streamvision`) needs `streamvision[webrtc,workflows]`; the
library parts (`streamvision.camera`, `streamvision.stream`, the TCP client and
entities) work without both extras; workflow pipelines need `workflows`.

```bash
STREAM_MANAGER_PORT=7070 python -m streamvision \
    --host-factory my_package.host:create_host \
    --host-setting api_key=... --warm-pipelines 1
```

`--host-factory` names a trusted callable returning a pipeline host; its module
is imported before the runtime. `scripts/streamvision_isolation_probe.py`
verifies an installed wheel runs without `inference`.

## Installation extras

| Extra | Needed for |
|---|---|
| `workflows` | `InferencePipeline.init_with_workflow`, `build_workflows_profiler`, the stream manager server, the WebRTC worker |
| `webrtc` | the stream manager server (`python -m streamvision`) and the WebRTC worker (`streamvision.webrtc_worker`) |
| `nvdec` | PyNvVideoCodec and Torch for `PyNvVideoCodecFrameProducer` (Linux x86_64 and Windows x64); ensure Torch is a compatible CUDA build |
| `test` | running the package tests |

The WebRTC worker (`streamvision.webrtc_worker`) requires `streamvision[webrtc,workflows]`.

Without `workflows`, cameras, `InferencePipeline.init_with_custom_logic`, sinks, the
watchdog and the stream manager client work. The two workflow functions raise
`CannotInitialiseModelError` with an install hint, and `python -m streamvision` exits
with a message naming the missing extra.

Without `workflows`, `concurrent.futures.Future` objects returned by custom logic reach
the sinks unresolved; with it they are resolved as before.
