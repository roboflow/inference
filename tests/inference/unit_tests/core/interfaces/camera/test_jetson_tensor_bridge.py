"""Unit coverage for the Python ABI boundary of the Jetson tensor bridge."""

import ctypes

from inference.core.interfaces.camera import jetson_tensor_bridge


class _FrameInfoLibrary:
    """Provide deterministic native frame metadata without loading CUDA code."""

    def rf_jetson_pipeline_get_frame_info(
        self, _handle, info_pointer, _error, _error_capacity
    ) -> int:
        """Populate the ABI struct as the native bridge would for one tensor."""

        info = ctypes.cast(
            info_pointer, ctypes.POINTER(jetson_tensor_bridge._FrameInfo)
        ).contents
        info.width = 1920
        info.height = 1080
        info.fps_numerator = 30000
        info.fps_denominator = 1001
        info.duration_ns = 10_000_000_000
        info.pts_ns = 3_003_000_000
        info.dts_ns = 2_969_633_333
        info.arrival_monotonic_ns = 9_000_000_000
        info.arrival_wall_time_ns = 1_700_000_000_000_000_000
        return 1


class _VersionLibrary:
    """Expose a fixed bridge version for compatibility tests."""

    def __init__(self, version: bytes) -> None:
        """Store the native ABI version returned by the fake library."""

        self._version = version

    def rf_jetson_tensor_bridge_version(self) -> bytes:
        """Return the simulated native ABI version."""

        return self._version


def test_frame_info_uses_metadata_for_the_grabbed_tensor() -> None:
    """Expose source and local timing without altering the tensor ownership API."""

    pipeline = jetson_tensor_bridge.NativeJetsonTensorPipeline.__new__(
        jetson_tensor_bridge.NativeJetsonTensorPipeline
    )
    pipeline._handle = ctypes.c_void_p(1)
    pipeline._library = _FrameInfoLibrary()

    frame_info = pipeline.frame_info()

    assert frame_info == jetson_tensor_bridge.JetsonFrameInfo(
        width=1920,
        height=1080,
        fps_numerator=30000,
        fps_denominator=1001,
        duration_ns=10_000_000_000,
        pts_ns=3_003_000_000,
        dts_ns=2_969_633_333,
        arrival_monotonic_ns=9_000_000_000,
        arrival_wall_time_ns=1_700_000_000_000_000_000,
    )


def test_frame_info_abi_includes_tensor_specific_timing() -> None:
    """Keep ctypes field order aligned with the versioned native struct."""

    assert [name for name, _ in jetson_tensor_bridge._FrameInfo._fields_] == [
        "width",
        "height",
        "fps_numerator",
        "fps_denominator",
        "duration_ns",
        "pts_ns",
        "dts_ns",
        "arrival_monotonic_ns",
        "arrival_wall_time_ns",
    ]
    assert ctypes.sizeof(jetson_tensor_bridge._FrameInfo) == 56


def test_bridge_probe_requires_timing_aware_abi(monkeypatch) -> None:
    """Refuse an older native library whose frame-info struct is shorter."""

    monkeypatch.setattr(
        jetson_tensor_bridge,
        "_load_bridge_library",
        lambda: _VersionLibrary(b"4"),
    )

    available, reason = jetson_tensor_bridge.jetson_tensor_bridge_available()

    assert not available
    assert "b'4'" in reason
