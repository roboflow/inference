"""Real decoder regressions using only synthetic media and a local HTTP canary."""

import struct
import threading
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from inference_models.utils import video


def _write_clip(path, *, container_format, codec, rate=24):
    with av.open(str(path), "w", format=container_format) as output:
        stream = output.add_stream(codec, rate=rate)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for index in range(24):
            pixels = np.full((48, 64, 3), index * 8, dtype=np.uint8)
            pixels[:24, :32] = (220, 20, 30)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _read_all(path):
    frames = []
    with video.open_video(path) as reader:
        while (frame := reader.read()) is not None:
            frames.append(frame)
    return frames


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "server-only.ts"
    _write_clip(path, container_format="mpegts", codec="mpeg2video")
    return path


def _hls(reference):
    return (
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n"
        f"#EXTINF:1.0,\n{reference}\n#EXT-X-ENDLIST\n"
    )


@pytest.mark.parametrize(
    "container_format,codec",
    [
        ("mp4", "mpeg4"),
        ("matroska", "mpeg4"),
        ("mpegts", "mpeg2video"),
        ("avi", "mpeg4"),
        ("webm", "libvpx-vp9"),
    ],
)
def test_self_contained_formats_match_sequential_opencv(
    tmp_path, container_format, codec
):
    path = tmp_path / "clip.video"
    _write_clip(path, container_format=container_format, codec=codec)
    with video.open_video(path) as reader:
        assert reader.fps == 24
        assert reader.frame_count == 24
    frames = _read_all(path)
    capture = cv2.VideoCapture(str(path))
    try:
        expected = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            expected.append(frame)
    finally:
        capture.release()

    assert len(frames) == len(expected) == 24
    for actual, original in zip(frames, expected):
        assert actual.shape == original.shape
        assert actual.dtype == np.uint8
        assert actual.flags.c_contiguous
        np.testing.assert_allclose(actual, original, atol=2)


@pytest.mark.parametrize("fps", [10, 60, Fraction(30000, 1001)])
def test_raw_h264_preserves_bitstream_frame_rate(tmp_path, fps):
    path = tmp_path / "clip.video"
    _write_clip(path, container_format="h264", codec="libx264", rate=fps)

    with video.open_video(path) as reader:
        assert reader.fps == pytest.approx(float(fps))

    assert len(_read_all(path)) == 24


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_display_rotation_matches_opencv(tmp_path, quarter_turns):
    path = tmp_path / "rotated.mp4"
    _write_clip(path, container_format="mp4", codec="mpeg4")
    data = bytearray(path.read_bytes())
    tkhd = data.index(b"tkhd")
    assert data[tkhd + 4] == 0  # version zero track header
    matrices = {
        1: (0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30),
        2: (-65536, 0, 0, 0, -65536, 0, 0, 0, 1 << 30),
        3: (0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30),
    }
    data[tkhd + 44 : tkhd + 80] = struct.pack(">9i", *matrices[quarter_turns])
    path.write_bytes(data)
    capture = cv2.VideoCapture(str(path))
    try:
        expected = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            expected.append(frame)
    finally:
        capture.release()

    actual = _read_all(path)
    assert len(actual) == len(expected) == 24
    for rotated, original in zip(actual, expected):
        assert rotated.shape == original.shape
        assert rotated.dtype == np.uint8
        assert rotated.flags.c_contiguous
        np.testing.assert_allclose(rotated, original, atol=2)


@pytest.mark.parametrize("suffix", [".m3u8", ".mp4", ".video"])
@pytest.mark.parametrize("kind", ["hls", "concat", "dash"])
def test_references_are_refused_regardless_of_extension(tmp_path, clip, suffix, kind):
    manifests = {
        "hls": _hls(clip),
        "concat": f"ffconcat version 1.0\nfile '{clip}'\n",
        "dash": (
            '<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011" '
            'type="static" mediaPresentationDuration="PT1S"><Period><AdaptationSet '
            'mimeType="video/mp2t"><Representation id="v" bandwidth="1000">'
            f"<BaseURL>{clip}</BaseURL></Representation></AdaptationSet></Period></MPD>"
        ),
    }
    path = tmp_path / ("upload" + suffix)
    path.write_text(manifests[kind])
    with pytest.raises(video.VideoDecodeError):
        _read_all(path)


def test_hls_fixture_really_decodes_the_secondary_file(tmp_path, clip):
    path = tmp_path / "playlist.m3u8"
    path.write_text(_hls(clip))
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 24


@pytest.mark.parametrize("allow_hls", [False, True])
def test_network_references_make_no_http_requests(
    tmp_path, clip, monkeypatch, allow_hls
):
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            payload = clip.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    # First establish a working canary; otherwise zero hits could hide a broken test.
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            path = tmp_path / "playlist.m3u8"
            path.write_text(_hls(f"http://127.0.0.1:{server.server_port}/clip.ts"))
            with av.open(
                str(path), options={"protocol_whitelist": "file,http,tcp"}
            ) as container:
                assert len(list(container.decode(video=0))) == 24
            assert hits
            hits.clear()
            if allow_hls:
                # Exercise the secondary-I/O guard independently of format rejection.
                monkeypatch.setattr(
                    video, "_VIDEO_DEMUXERS", video._VIDEO_DEMUXERS + ",hls"
                )
            with pytest.raises(video.VideoDecodeError):
                _read_all(path)
            assert hits == []
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_secondary_file_guard_even_if_manifest_format_is_allowed(
    tmp_path, clip, monkeypatch
):
    path = tmp_path / "playlist.m3u8"
    path.write_text(_hls(clip))
    monkeypatch.setattr(video, "_VIDEO_DEMUXERS", video._VIDEO_DEMUXERS + ",hls")
    with pytest.raises(video.VideoDecodeError, match="referencing other files"):
        _read_all(path)


def test_primary_file_closes_on_rejection_and_early_exit(tmp_path, clip, monkeypatch):
    opened = []
    real_open = Path.open

    def record_open(path, *args, **kwargs):
        source = real_open(path, *args, **kwargs)
        opened.append(source)
        return source

    invalid = tmp_path / "bad.mp4"
    invalid.write_bytes(b"not a video")
    monkeypatch.setattr(Path, "open", record_open)
    with pytest.raises(video.VideoDecodeError):
        _read_all(invalid)
    with video.open_video(clip) as reader:
        assert reader.read() is not None
    assert len(opened) == 2
    assert all(source.closed for source in opened)


def test_variable_frame_rate_keeps_decode_order_and_average_rate(tmp_path):
    from fractions import Fraction

    path = tmp_path / "variable.mp4"
    time_base = Fraction(1, 24)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=24)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = time_base
        pts = 0
        for index in range(24):
            pixels = np.full((48, 64, 3), index * 8, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.time_base, frame.pts = time_base, pts
            pts += 1 + index % 2
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    capture = cv2.VideoCapture(str(path))
    try:
        expected_fps = capture.get(cv2.CAP_PROP_FPS)
        with video.open_video(path) as reader:
            assert reader.fps == pytest.approx(expected_fps)
            assert reader.frame_count == 24
            assert reader.fps != 24  # The fixture actually has variable timestamps.
            for _ in range(24):
                ok, expected = capture.read()
                assert ok
                np.testing.assert_allclose(reader.read(), expected, atol=2)
            assert reader.read() is None
    finally:
        capture.release()
