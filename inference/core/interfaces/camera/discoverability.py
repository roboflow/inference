"""Runtime discovery of hardware video frame producers."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from inference.core.interfaces.camera.entities import VideoFrameProducer
from inference.core.logger import logger

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
    gst_ok, gst_reason = probe_gstreamer_elements(required_gstreamer_elements(video))
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
        JETSON: check_jetson_gstreamer(video=video),
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
    for name in _resolution_order(prefer, video, output_tensor=output_tensor):
        if not checks[name].available:
            continue
        if name in (GSTREAMER_CUDA, DGPU):
            # The onnx.gpu image's CUDA media stack is built for CC>=7.5; fail fast
            # with a clear message on older GPUs instead of a cryptic runtime crash.
            # (Raised here so it propagates to the caller, which logs it and falls
            # back to the cv2 CPU decoder.)
            _require_media_compute_capability()
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
        ) as error:  # noqa: BLE001 - probe said ok but construction failed; try next
            # Without this log a hardware-backend failure is invisible: the
            # probe result still reads "ok" and the source lands on the cv2
            # CPU path with no trace of why.
            logger.warning(
                "Hardware video producer %s failed to construct for %r: %r. "
                "Trying the next candidate.",
                name,
                video,
                error,
            )
            continue
    return None


def _resolution_order(
    prefer: Optional[str],
    video: Optional[Union[str, int]] = None,
    *,
    output_tensor: bool = True,
) -> List[str]:
    """Source-type-aware producer routing (per the media-decode plan).

    An explicit ``prefer`` always wins, with the remaining backends kept as
    fallbacks. Otherwise the backend is chosen from (platform, source type):

    - **Jetson** (aarch64 Linux): GStreamer for live/stream/camera sources and for
      local files when CUDA tensor output is requested. Numpy file consumers retain
      the cv2 CPU decoder fallback.
    - **dGPU / x86**: GStreamer for live/stream sources; PyNvVideoCodec (``dgpu``) for
      local FILES (its ``SimpleDecoder`` is seekable-file only).

    ``http(s)`` and other URI schemes are treated as streams (GStreamer handles them;
    PyNvVideoCodec cannot seek them).
    """
    producer_names = (GSTREAMER_CUDA, JETSON, DGPU)
    if prefer in producer_names:
        return [prefer] + [name for name in producer_names if name != prefer]
    import platform

    is_file = _is_file_source(video)
    if platform.machine() == "aarch64" and platform.system() == "Linux":
        # Jetson files are lossless in the tensor bridge. Keep the established
        # cv2 route for CPU/Numpy callers, which do not benefit from CUDA output.
        return [JETSON] if output_tensor or not is_file else []
    # dGPU / x86: streams -> GStreamer; local files -> PyNvVideoCodec.
    return [DGPU] if is_file else [GSTREAMER_CUDA]


def _is_file_source(video: Optional[Union[str, int]]) -> bool:
    """A seekable local FILE path (routed to opencv on Jetson / PyNvVideoCodec on
    dGPU), as opposed to a live/stream/camera source (routed to GStreamer). URI
    schemes (rtsp/http/...), ``/dev/video*``, ``csi://`` and integer camera indices
    are NOT files."""
    return (
        isinstance(video, str)
        and "://" not in video
        and not video.startswith("/dev/video")
        and not video.lower().startswith("csi://")
    )


def _require_media_compute_capability(minimum: tuple = (7, 5)) -> None:
    """Guard the dGPU CUDA media path (GStreamer-CUDA / PyNvVideoCodec) against GPUs
    below the compiled compute-capability floor.

    ``Dockerfile.onnx.gpu`` builds OpenCV-CUDA + nvcodec for CC>=7.5 only, so on
    older GPUs (V100 7.0, Pascal 6.x) the nvcodec conversion kernels and cv2.cuda ops
    fail at runtime with an opaque 'no kernel image' error. Detect this at producer
    selection and raise a clear, actionable error; the caller (``VideoSource``) logs
    it and falls back to the cv2 CPU decode path.

    Robust in non-CUDA / test environments: if CUDA is absent or the capability can't
    be read, this does nothing (the ``check_*`` probe already gates on CUDA, and the
    runtime would surface any real issue)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        capability = tuple(torch.cuda.get_device_capability())
    except Exception:  # noqa: BLE001 - can't determine CC -> don't block here
        return
    if capability < minimum:
        raise RuntimeError(
            f"Hardware GPU video decode requires CUDA compute capability "
            f">= {minimum[0]}.{minimum[1]}, but this GPU is "
            f"{capability[0]}.{capability[1]}. The onnx.gpu image's OpenCV-CUDA + "
            f"nvcodec media stack is compiled for >= {minimum[0]}.{minimum[1]} only; "
            f"use the CPU (cv2) decode path on this hardware."
        )


def _dgpu_supports_source(video: Optional[Union[str, int]]) -> bool:
    if video is None:
        return True
    return (
        isinstance(video, str)
        and "://" not in video
        and not video.startswith("/dev/video")
    )
