import ctypes
import ctypes.util
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from inference.core.interfaces.camera.exceptions import NativeGrabTimeoutError

_ERROR_CAPACITY = 1024
_INFINITE_TIMEOUT_NS = (1 << 64) - 1
_GRAB_STATUS_TIMEOUT = 2
_DEFAULT_LIBRARY_PATH = "/opt/roboflow/lib/libroboflow_gstreamer_cuda_tensor.so.1"


class _FrameInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("fps_numerator", ctypes.c_int32),
        ("fps_denominator", ctypes.c_int32),
        ("duration_ns", ctypes.c_int64),
    ]


class _BridgeStats(ctypes.Structure):
    _fields_ = [
        ("frames", ctypes.c_uint64),
        ("cuda_maps", ctypes.c_uint64),
        ("host_pixel_maps", ctypes.c_uint64),
        ("host_to_device_copies", ctypes.c_uint64),
        ("device_to_host_copies", ctypes.c_uint64),
        ("stream_synchronizations", ctypes.c_uint64),
        ("active_leases", ctypes.c_uint64),
        ("last_channel_stride", ctypes.c_int64),
        ("last_row_stride", ctypes.c_int64),
    ]


@dataclass(frozen=True)
class GstreamerCudaFrameInfo:
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    duration_ns: int


def gstreamer_cuda_tensor_bridge_available() -> Tuple[bool, str]:
    try:
        library = _load_bridge_library()
        version = library.rf_gstreamer_cuda_tensor_bridge_version()
    except Exception as error:  # noqa: BLE001 - runtime capability probe
        return False, f"GStreamer CUDA tensor bridge is unavailable: {error!r}"
    if version != b"3":
        return False, f"Unsupported GStreamer CUDA tensor bridge version: {version!r}"
    return True, "ok"


class NativeGstreamerCudaTensorPipeline:
    def __init__(self, pipeline: str, *, device_id: int = 0) -> None:
        # Set before any statement that can raise so __del__ -> close() works
        # on a partially constructed object.
        self._teardown_lock = threading.Lock()
        self._library = _load_bridge_library()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        self._handle = self._library.rf_gstreamer_cuda_pipeline_create(
            pipeline.encode("utf-8"),
            device_id,
            error,
            len(error),
        )
        if not self._handle:
            raise RuntimeError(_decode_error(error))

    def grab(self, timeout_ns: Optional[int] = None) -> bool:
        self._ensure_open()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = self._library.rf_gstreamer_cuda_pipeline_grab(
            self._handle,
            _INFINITE_TIMEOUT_NS if timeout_ns is None else timeout_ns,
            error,
            len(error),
        )
        if status == _GRAB_STATUS_TIMEOUT:
            raise NativeGrabTimeoutError(
                "GStreamer CUDA pipeline did not produce a frame within the timeout"
            )
        if status < 0:
            raise RuntimeError(_decode_error(error))
        return status == 1

    def retrieve(self):
        self._ensure_open()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        managed_tensor = self._library.rf_gstreamer_cuda_pipeline_retrieve(
            self._handle,
            error,
            len(error),
        )
        if not managed_tensor:
            raise RuntimeError(_decode_error(error))

        capsule = _create_dlpack_capsule(managed_tensor)
        try:
            import torch

            tensor = torch.utils.dlpack.from_dlpack(capsule)
        except Exception:
            # torch renames a consumed capsule to "used_dltensor"; only free
            # the tensor if the capsule still owns it, otherwise this would
            # double-free.
            if _capsule_owns_tensor(capsule):
                self._library.rf_gstreamer_cuda_dlpack_delete(managed_tensor)
            raise
        if not tensor.is_cuda or tensor.dtype != torch.uint8 or tensor.ndim != 3:
            raise RuntimeError(
                "GStreamer CUDA bridge returned an invalid CUDA uint8 CHW tensor"
            )
        return tensor

    def frame_info(self) -> GstreamerCudaFrameInfo:
        self._ensure_open()
        info = _FrameInfo()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        status = self._library.rf_gstreamer_cuda_pipeline_get_frame_info(
            self._handle,
            ctypes.byref(info),
            error,
            len(error),
        )
        if status < 0:
            raise RuntimeError(_decode_error(error))
        return GstreamerCudaFrameInfo(
            width=info.width,
            height=info.height,
            fps_numerator=info.fps_numerator,
            fps_denominator=info.fps_denominator,
            duration_ns=info.duration_ns,
        )

    def has_factory(self, factory_name: str) -> bool:
        self._ensure_open()
        return bool(
            self._library.rf_gstreamer_cuda_pipeline_has_factory(
                self._handle, factory_name.encode("utf-8")
            )
        )

    def stats(self) -> Dict[str, int]:
        self._ensure_open()
        stats = _BridgeStats()
        status = self._library.rf_gstreamer_cuda_pipeline_get_stats(
            self._handle, ctypes.byref(stats)
        )
        if status < 0:
            raise RuntimeError("Could not read GStreamer CUDA bridge statistics")
        return {name: int(getattr(stats, name)) for name, _ in stats._fields_}

    def interrupt(self) -> None:
        # Serialized with close(): a concurrent release frees the native
        # handle, and interrupting a freed handle is a use-after-free. grab()
        # must stay outside this lock; it blocks and interrupt() unblocks it.
        with self._teardown_lock:
            handle = getattr(self, "_handle", None)
            if handle:
                status = self._library.rf_gstreamer_cuda_pipeline_interrupt(handle)
                if status < 0:
                    raise RuntimeError("Could not interrupt GStreamer CUDA pipeline")

    def close(self) -> None:
        with self._teardown_lock:
            handle = getattr(self, "_handle", None)
            if handle:
                self._handle = None
                self._library.rf_gstreamer_cuda_pipeline_release(handle)

    def _ensure_open(self) -> None:
        if not getattr(self, "_handle", None):
            raise RuntimeError("GStreamer CUDA tensor pipeline is closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


def _load_bridge_library():
    configured_path = os.getenv("ROBOFLOW_GSTREAMER_CUDA_TENSOR_BRIDGE_LIBRARY")
    candidates = [configured_path] if configured_path else []
    candidates.extend(
        [
            _DEFAULT_LIBRARY_PATH,
            ctypes.util.find_library("roboflow_gstreamer_cuda_tensor"),
        ]
    )
    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
            _configure_library(library)
            return library
        except OSError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise OSError("GStreamer CUDA tensor bridge library was not found")


def _configure_library(library) -> None:
    library.rf_gstreamer_cuda_tensor_bridge_version.argtypes = []
    library.rf_gstreamer_cuda_tensor_bridge_version.restype = ctypes.c_char_p
    library.rf_gstreamer_cuda_pipeline_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.rf_gstreamer_cuda_pipeline_create.restype = ctypes.c_void_p
    library.rf_gstreamer_cuda_pipeline_grab.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.rf_gstreamer_cuda_pipeline_grab.restype = ctypes.c_int
    library.rf_gstreamer_cuda_pipeline_get_frame_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_FrameInfo),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.rf_gstreamer_cuda_pipeline_get_frame_info.restype = ctypes.c_int
    library.rf_gstreamer_cuda_pipeline_has_factory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    library.rf_gstreamer_cuda_pipeline_has_factory.restype = ctypes.c_int
    library.rf_gstreamer_cuda_pipeline_retrieve.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.rf_gstreamer_cuda_pipeline_retrieve.restype = ctypes.c_void_p
    library.rf_gstreamer_cuda_pipeline_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_BridgeStats),
    ]
    library.rf_gstreamer_cuda_pipeline_get_stats.restype = ctypes.c_int
    library.rf_gstreamer_cuda_pipeline_interrupt.argtypes = [ctypes.c_void_p]
    library.rf_gstreamer_cuda_pipeline_interrupt.restype = ctypes.c_int
    library.rf_gstreamer_cuda_dlpack_delete.argtypes = [ctypes.c_void_p]
    library.rf_gstreamer_cuda_dlpack_delete.restype = None
    library.rf_gstreamer_cuda_pipeline_release.argtypes = [ctypes.c_void_p]
    library.rf_gstreamer_cuda_pipeline_release.restype = None


def _create_dlpack_capsule(managed_tensor):
    py_capsule_new = ctypes.pythonapi.PyCapsule_New
    py_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    py_capsule_new.restype = ctypes.py_object
    return py_capsule_new(managed_tensor, b"dltensor", None)


def _capsule_owns_tensor(capsule) -> bool:
    py_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
    py_capsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    py_capsule_is_valid.restype = ctypes.c_int
    return bool(py_capsule_is_valid(capsule, b"dltensor"))


def _decode_error(error_buffer) -> str:
    message = error_buffer.value.decode("utf-8", errors="replace")
    return message or "GStreamer CUDA tensor bridge failed"
