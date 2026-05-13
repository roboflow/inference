"""Unit tests for WebRTC codec preference helpers."""

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from inference_sdk.webrtc.codec_preferences import (
    _order_codecs_by_mime_type,
    apply_video_codec_preference,
)
from inference_sdk.webrtc.config import StreamConfig


@dataclass(frozen=True, eq=False)
class _Codec:
    mimeType: str


class _Transceiver:
    def __init__(self, kind: str):
        self.kind = kind
        self.codec_preferences = None

    def setCodecPreferences(self, codecs):
        self.codec_preferences = codecs


class _PeerConnection:
    def __init__(self, transceivers):
        self._transceivers = transceivers

    def getTransceivers(self):
        return self._transceivers


def test_stream_config_defaults_to_h264():
    assert StreamConfig().video_codec == "h264"


def test_stream_config_allows_auto_codec_negotiation():
    assert StreamConfig(video_codec="auto").video_codec == "auto"


def test_stream_config_rejects_unknown_video_codec():
    with pytest.raises(ValueError, match="video_codec must be one of"):
        StreamConfig(video_codec="h265")


def test_order_codecs_by_mime_type_preserves_fallback_codecs():
    vp8 = _Codec("video/VP8")
    h264_baseline = _Codec("video/H264")
    rtx = _Codec("video/rtx")
    h264_high = _Codec("video/H264")

    ordered = _order_codecs_by_mime_type(
        [vp8, h264_baseline, rtx, h264_high],
        "video/h264",
    )

    assert ordered == [h264_baseline, h264_high, vp8, rtx]


def test_apply_video_codec_preference_sets_video_transceivers(monkeypatch):
    vp8 = _Codec("video/VP8")
    h264 = _Codec("video/H264")
    rtx = _Codec("video/rtx")

    class RTCRtpSender:
        @staticmethod
        def getCapabilities(kind):
            assert kind == "video"
            return SimpleNamespace(codecs=[vp8, h264, rtx])

    aiortc = ModuleType("aiortc")
    aiortc.RTCRtpSender = RTCRtpSender
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)

    video = _Transceiver("video")
    audio = _Transceiver("audio")
    pc = _PeerConnection([video, audio])

    assert apply_video_codec_preference(pc, "h264") is True
    assert video.codec_preferences == [h264, vp8, rtx]
    assert audio.codec_preferences is None
