import sys
from types import SimpleNamespace

from inference.core.interfaces.webrtc_worker.aiortc_bitrate import (
    H264_BITRATE_LIMITS_BPS,
    REMB_MAX_BITRATE_BPS,
    REMB_MIN_BITRATE_BPS,
    REMB_START_BITRATE_BPS,
    _apply_codec_bitrate_limits,
    _clamp_bitrate,
    _patch_receiver_remb_estimator,
    prefer_h264_for_peer_connection,
)


def test_apply_codec_bitrate_limits_sets_aiortc_constants():
    module = SimpleNamespace(
        DEFAULT_BITRATE=1,
        MIN_BITRATE=1,
        MAX_BITRATE=1,
    )

    applied = _apply_codec_bitrate_limits(module, H264_BITRATE_LIMITS_BPS)

    assert applied == H264_BITRATE_LIMITS_BPS
    assert module.DEFAULT_BITRATE == H264_BITRATE_LIMITS_BPS["DEFAULT_BITRATE"]
    assert module.MIN_BITRATE == H264_BITRATE_LIMITS_BPS["MIN_BITRATE"]
    assert module.MAX_BITRATE == H264_BITRATE_LIMITS_BPS["MAX_BITRATE"]


def test_clamp_bitrate_applies_min_and_max():
    assert (
        _clamp_bitrate(
            REMB_MIN_BITRATE_BPS - 1,
            min_bitrate=REMB_MIN_BITRATE_BPS,
            max_bitrate=REMB_MAX_BITRATE_BPS,
        )
        == REMB_MIN_BITRATE_BPS
    )
    assert (
        _clamp_bitrate(
            REMB_MAX_BITRATE_BPS + 1,
            min_bitrate=REMB_MIN_BITRATE_BPS,
            max_bitrate=REMB_MAX_BITRATE_BPS,
        )
        == REMB_MAX_BITRATE_BPS
    )


def test_receiver_remb_estimator_is_seeded_and_clamped():
    class FakeRateControl:
        def __init__(self):
            self.current_bitrate = None
            self.latest_estimated_throughput = None

        def set_estimate(self, bitrate, now_ms):
            self.current_bitrate = bitrate

    class FakeRemoteBitrateEstimator:
        def __init__(self):
            self.rate_control = FakeRateControl()

        def add(self, arrival_time_ms, abs_send_time, payload_size, ssrc):
            return REMB_MIN_BITRATE_BPS - 1, [ssrc]

    rate_module = SimpleNamespace(RemoteBitrateEstimator=FakeRemoteBitrateEstimator)

    _patch_receiver_remb_estimator(rate_module)
    estimator = rate_module.RemoteBitrateEstimator()

    assert estimator.rate_control.current_bitrate == REMB_START_BITRATE_BPS
    assert estimator.rate_control.latest_estimated_throughput == REMB_START_BITRATE_BPS
    assert estimator.add(1, 1, 1, 1234) == (REMB_MIN_BITRATE_BPS, [1234])
    assert estimator.rate_control.current_bitrate == REMB_MIN_BITRATE_BPS


def test_prefer_h264_for_peer_connection_keeps_other_codecs_as_fallback(monkeypatch):
    h264 = SimpleNamespace(mimeType="video/H264")
    vp8 = SimpleNamespace(mimeType="video/VP8")
    av1 = SimpleNamespace(mimeType="video/AV1")

    class FakeSender:
        @staticmethod
        def getCapabilities(kind):
            assert kind == "video"
            return SimpleNamespace(codecs=[vp8, h264, av1])

    class FakeTransceiver:
        kind = "video"

        def __init__(self):
            self.preferred_codecs = None

        def setCodecPreferences(self, codecs):
            self.preferred_codecs = codecs

    transceiver = FakeTransceiver()
    peer_connection = SimpleNamespace(getTransceivers=lambda: [transceiver])
    monkeypatch.setitem(sys.modules, "aiortc", SimpleNamespace(RTCRtpSender=FakeSender))

    assert prefer_h264_for_peer_connection(peer_connection) is True
    assert transceiver.preferred_codecs == [h264, vp8, av1]
