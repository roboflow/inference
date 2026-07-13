"""Runtime discovery of hardware video frame producers."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from inference.core.interfaces.camera.entities import VideoFrameProducer

JETSON = "jetson"
DGPU = "dgpu"
GSTREAMER_CUDA = "gstreamer_cuda"


@dataclass(frozen=True)
class ProducerAvailability:
    """Result of probing one hardware-decode backend in this environment."""

    name: str
    available: bool
    reason: str


def check_jetson_gstreamer(
    video: Optional[Union[str, int]] = None,
    *,
    require_cuda_tensor: bool = True,
) -> ProducerAvailability:
    """Probe the in-repo Jetson GStreamer producer and its required elements."""

    try:
        from inference.core.interfaces.camera.jetson_producer import (
            probe_gstreamer_elements,
            required_gstreamer_elements,
        )
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(
            JETSON, False, f"Jetson GStreamer producer import failed: {error!r}"
        )
    gst_ok, gst_reason = probe_gstreamer_elements(
        required_gstreamer_elements(video, output_tensor=True)
    )
    if not gst_ok:
        return ProducerAvailability(JETSON, False, gst_reason)
    try:
        from inference.core.interfaces.camera.jetson_tensor_bridge import (
            jetson_tensor_bridge_available,
        )

        bridge_ok, bridge_reason = jetson_tensor_bridge_available()
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(
            JETSON, False, f"Jetson tensor bridge probe failed: {error!r}"
        )
    if not bridge_ok:
        return ProducerAvailability(JETSON, False, bridge_reason)
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(JETSON, False, f"torch import failed: {error!r}")
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(
            JETSON, False, f"torch.cuda probe failed: {error!r}"
        )
    if not cuda_ok:
        return ProducerAvailability(JETSON, False, "torch.cuda.is_available() is False")
    return ProducerAvailability(JETSON, True, "ok")


def check_pynvvideocodec() -> ProducerAvailability:
    """Probe whether the dGPU (``PyNvVideoCodec``) producer is usable here."""
    try:
        import PyNvVideoCodec  # noqa: F401
    except Exception as error:  # noqa: BLE001
        # Common failure: `libnvidia-encode.so.1: cannot open shared object file` — the
        # NVENC lib that PyNvVideoCodec hard-links even for decode. Treated as unavailable.
        return ProducerAvailability(
            DGPU, False, f"PyNvVideoCodec import failed: {error!r}"
        )
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(DGPU, False, f"torch import failed: {error!r}")
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(DGPU, False, f"torch.cuda probe failed: {error!r}")
    if not cuda_ok:
        return ProducerAvailability(DGPU, False, "torch.cuda.is_available() is False")
    return ProducerAvailability(DGPU, True, "ok")


def check_gstreamer_cuda(
    video: Optional[Union[str, int]] = None,
) -> ProducerAvailability:
    if video is not None and (
        not isinstance(video, str)
        or video.startswith("/dev/video")
        or video.lower().startswith("csi://")
    ):
        return ProducerAvailability(
            GSTREAMER_CUDA,
            False,
            "GStreamer CUDA producer requires a URI or file path",
        )
    try:
        from inference.core.interfaces.camera.gstreamer_cuda_producer import (
            probe_gstreamer_cuda_elements,
            required_gstreamer_cuda_elements,
        )
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(
            GSTREAMER_CUDA,
            False,
            f"GStreamer CUDA producer import failed: {error!r}",
        )
    gst_ok, gst_reason = probe_gstreamer_cuda_elements(
        required_gstreamer_cuda_elements(video)
    )
    if not gst_ok:
        return ProducerAvailability(GSTREAMER_CUDA, False, gst_reason)
    try:
        from inference.core.interfaces.camera.gstreamer_cuda_tensor_bridge import (
            gstreamer_cuda_tensor_bridge_available,
        )

        bridge_ok, bridge_reason = gstreamer_cuda_tensor_bridge_available()
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(
            GSTREAMER_CUDA,
            False,
            f"GStreamer CUDA tensor bridge probe failed: {error!r}",
        )
    if not bridge_ok:
        return ProducerAvailability(GSTREAMER_CUDA, False, bridge_reason)
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(
            GSTREAMER_CUDA, False, f"torch import failed: {error!r}"
        )
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(
            GSTREAMER_CUDA, False, f"torch.cuda probe failed: {error!r}"
        )
    if not cuda_ok:
        return ProducerAvailability(
            GSTREAMER_CUDA, False, "torch.cuda.is_available() is False"
        )
    return ProducerAvailability(GSTREAMER_CUDA, True, "ok")


def available_producers(
    video: Optional[Union[str, int]] = None,
    *,
    require_cuda_tensor: bool = True,
) -> Dict[str, ProducerAvailability]:
    """Probe every hardware-decode backend; map name -> :class:`ProducerAvailability`."""
    if require_cuda_tensor and _dgpu_supports_source(video):
        dgpu_availability = check_pynvvideocodec()
    elif not require_cuda_tensor:
        dgpu_availability = ProducerAvailability(
            DGPU, False, "PyNvVideoCodec produces tensor frames"
        )
    else:
        dgpu_availability = ProducerAvailability(
            DGPU, False, "PyNvVideoCodec SimpleDecoder requires a file source"
        )
    gstreamer_cuda_availability = check_gstreamer_cuda(video)
    return {
        GSTREAMER_CUDA: gstreamer_cuda_availability,
        JETSON: check_jetson_gstreamer(
            video=video, require_cuda_tensor=require_cuda_tensor
        ),
        DGPU: dgpu_availability,
    }


def build_hw_producer(
    video: Union[str, int],
    *,
    prefer: Optional[str] = None,
    output_tensor: bool = True,
    **producer_kwargs,
) -> Optional[VideoFrameProducer]:
    """Best-effort factory: return an instantiated GPU producer for ``video``, or ``None``.

    Resolution order: ``prefer`` (``"jetson"``/``"dgpu"``) first if given; otherwise
    Jetson is preferred on ``aarch64`` Linux, dGPU elsewhere. Only backends that pass their
    ``check_*`` probe are attempted, and instantiation failures fall through to the next
    candidate. The producer modules are imported locally so this stays import-safe.

    Source capability is included in discovery. The PyNvVideoCodec backend accepts file
    references, while the Jetson GStreamer backend accepts files, cameras, and network
    streams.
    """
    checks = available_producers(
        video=video,
        require_cuda_tensor=output_tensor,
    )
    for name in _resolution_order(prefer):
        if not checks[name].available:
            continue
        try:
            if name == GSTREAMER_CUDA:
                from inference.core.interfaces.camera.gstreamer_cuda_producer import (
                    GstreamerCudaVideoFrameProducer,
                )

                return GstreamerCudaVideoFrameProducer(
                    video,
                    output_tensor=output_tensor,
                    **producer_kwargs,
                )
            if name == JETSON:
                from inference.core.interfaces.camera.jetson_producer import (
                    JetsonVideoFrameProducer,
                )

                return JetsonVideoFrameProducer(
                    video,
                    output_tensor=output_tensor,
                    **producer_kwargs,
                )
            if name == DGPU and output_tensor:
                from inference.core.interfaces.camera.dgpu_producer import (
                    PyNvVideoCodecFrameProducer,
                )

                return PyNvVideoCodecFrameProducer(video, **producer_kwargs)
        except (
            Exception
        ):  # noqa: BLE001 - probe said ok but construction failed; try next
            continue
    return None


def _resolution_order(prefer: Optional[str]) -> List[str]:
    producer_names = (GSTREAMER_CUDA, JETSON, DGPU)
    if prefer in producer_names:
        return [prefer] + [name for name in producer_names if name != prefer]
    import platform

    if platform.machine() == "aarch64" and platform.system() == "Linux":
        return [JETSON, GSTREAMER_CUDA, DGPU]
    return [GSTREAMER_CUDA, DGPU, JETSON]


def _dgpu_supports_source(video: Optional[Union[str, int]]) -> bool:
    if video is None:
        return True
    return (
        isinstance(video, str)
        and "://" not in video
        and not video.startswith("/dev/video")
    )
