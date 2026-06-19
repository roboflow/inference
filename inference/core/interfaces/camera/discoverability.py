"""Runtime discovery + verification of the optional hardware-decode ``VideoFrameProducer``s.

These producers depend on libraries that are not always installed:
- ``jetson_utils`` ships only with JetPack (Jetson devices), and
- ``PyNvVideoCodec`` needs a discrete NVIDIA GPU + the Video Codec SDK (and it hard-links
  ``libnvidia-encode`` even for decode, so it fails to import on NVENC-less GPUs / on
  containers without the ``video`` driver capability).

Every dependency import here is **local** (inside a function) and guarded, so importing
this module never fails. Call the ``check_*`` functions at runtime to decide what's usable
in the current environment, then build the matching producer with :func:`build_hw_producer`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from inference.core.interfaces.camera.entities import VideoFrameProducer

JETSON = "jetson"
DGPU = "dgpu"


@dataclass(frozen=True)
class ProducerAvailability:
    """Result of probing one hardware-decode backend in this environment."""

    name: str
    available: bool
    reason: str


def check_jetson_utils() -> ProducerAvailability:
    """Probe whether the Jetson (``jetson_utils``) producer is usable here."""
    try:
        import jetson_utils  # noqa: F401
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(JETSON, False, f"jetson_utils import failed: {error!r}")
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(JETSON, False, f"torch import failed: {error!r}")
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as error:  # noqa: BLE001
        return ProducerAvailability(JETSON, False, f"torch.cuda probe failed: {error!r}")
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


def available_producers() -> Dict[str, ProducerAvailability]:
    """Probe every hardware-decode backend; map name -> :class:`ProducerAvailability`."""
    return {
        JETSON: check_jetson_utils(),
        DGPU: check_pynvvideocodec(),
    }


def build_hw_producer(
    video: str,
    *,
    prefer: Optional[str] = None,
    **producer_kwargs,
) -> Optional[VideoFrameProducer]:
    """Best-effort factory: return an instantiated GPU producer for ``video``, or ``None``.

    Resolution order: ``prefer`` (``"jetson"``/``"dgpu"``) first if given; otherwise
    Jetson is preferred on ``aarch64`` Linux, dGPU elsewhere. Only backends that pass their
    ``check_*`` probe are attempted, and instantiation failures fall through to the next
    candidate. The producer modules are imported locally so this stays import-safe.

    NOTE: this only matches *installed-and-working* backends; it does not yet reason about
    whether the backend supports the submitted input *type* (e.g. PyNvVideoCodec is
    file-only — an rtsp:// URI should skip it). Add that input-capability gate when wiring
    live sources.
    """
    checks = available_producers()
    for name in _resolution_order(prefer):
        if not checks[name].available:
            continue
        try:
            if name == JETSON:
                from inference.core.interfaces.camera.jetson_producer import (
                    JetsonVideoFrameProducer,
                )

                return JetsonVideoFrameProducer(video, **producer_kwargs)
            if name == DGPU:
                from inference.core.interfaces.camera.dgpu_producer import (
                    PyNvVideoCodecFrameProducer,
                )

                return PyNvVideoCodecFrameProducer(video, **producer_kwargs)
        except Exception:  # noqa: BLE001 - probe said ok but construction failed; try next
            continue
    return None


def _resolution_order(prefer: Optional[str]) -> List[str]:
    if prefer in (JETSON, DGPU):
        return [prefer] + [n for n in (JETSON, DGPU) if n != prefer]
    import platform

    if platform.machine() == "aarch64" and platform.system() == "Linux":
        return [JETSON, DGPU]
    return [DGPU, JETSON]
