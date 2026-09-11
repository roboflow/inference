"""Open MJPEG over validated HTTP connections, never through FFmpeg networking."""

import time
from contextlib import ExitStack
from urllib.parse import urljoin, urlsplit

import requests
from aiortc.contrib.media import MediaPlayer

from inference.core.interfaces.camera.source_reference_sanitizer import (
    sanitize_source_reference,
)
from inference.core.utils.url_input import SSRFProtectedHTTPAdapter

MJPEG_TIMEOUT_SECONDS = 5.0
MJPEG_MAX_REDIRECTS = 5


class _MJPEGReader:
    def __init__(self, response: requests.Response):
        self.response = response
        self.open_deadline = time.monotonic() + MJPEG_TIMEOUT_SECONDS

    def read(self, size: int) -> bytes:
        try:
            if self.open_deadline and time.monotonic() >= self.open_deadline:
                raise TimeoutError("MJPEG opening timed out")
            # Return available bytes so PyAV can check its timeout between reads.
            # Filling an entire buffer lets a trickling peer evade read timeouts.
            read1 = getattr(self.response.raw, "read1", None)
            if read1 is not None:
                return read1(size)
            # ponytail: urllib3 1.x reads one byte; upgrade to 2.x for buffered read1.
            return self.response.raw.read(min(size, 1))
        except Exception:
            # HTTP errors can embed credentials, including during playback.
            raise RuntimeError(
                "Could not read MJPEG stream (failed or timed out)"
            ) from None


def open_mjpeg_player(
    url: str, *, allow_non_global_addresses: bool = False
) -> MediaPlayer:
    """Validate/pin each HTTP hop and keep its response alive until the track ends."""
    try:
        with ExitStack() as resources:
            adapter = SSRFProtectedHTTPAdapter(
                allow_non_global_addresses=allow_non_global_addresses
            )
            resources.callback(adapter.close)
            for redirect in range(MJPEG_MAX_REDIRECTS + 1):
                parsed = urlsplit(url)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or "\\" in parsed.netloc
                ):
                    raise ValueError("MJPEG requires an HTTP(S) URL with a valid host")
                request = requests.Request("GET", url).prepare()
                # Send through the adapter directly: no environment proxies/netrc
                # and no automatic redirects that drain unbounded response bodies.
                response = adapter.send(
                    request, stream=True, timeout=MJPEG_TIMEOUT_SECONDS, proxies={}
                )
                if response.is_redirect:
                    response.close()
                    if redirect == MJPEG_MAX_REDIRECTS:
                        raise ValueError("Too many MJPEG redirects")
                    url = urljoin(request.url, response.headers["Location"])
                    continue
                resources.callback(response.close)
                response.raise_for_status()
                reader = _MJPEGReader(response)
                content_type = response.headers.get("Content-Type", "").lower()
                player = MediaPlayer(
                    reader,
                    format=(
                        "mpjpeg"
                        if content_type.startswith("multipart/x-mixed-replace")
                        else "mjpeg"
                    ),
                    options={"protocol_whitelist": ""},
                    timeout=MJPEG_TIMEOUT_SECONDS,
                )
                if player.video is None:
                    raise ValueError("MJPEG stream has no video track")
                reader.open_deadline = None
                # PlayerStreamTrack.stop emits ended on both shutdown and EOF.
                player.video.on("ended", resources.pop_all().close)
                return player
    except Exception as error:
        # Transport errors can include a credentialed relative request target.
        raise RuntimeError(
            f"Failed to open MJPEG stream {sanitize_source_reference(url)}: "
            f"{type(error).__name__}"
        ) from None
