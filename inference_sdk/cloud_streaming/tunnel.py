"""RTSPS with a verified server certificate.

FFmpeg's RTSPS client does not verify the server certificate, and its RTSP layer offers no way to
ask it to, so an `rtsps://` url handed to FFmpeg would send video and the publish credential to
whoever answers. FFmpeg therefore speaks plain RTSP to a tunnel on the loopback interface, and the
tunnel speaks TLS to the relay, verified against the system trust store and the relay's hostname.

The tunnel runs as a child process: PyAV holds the GIL through FFmpeg's RTSP handshake, so a
tunnel thread in the same interpreter could never answer it. The child uses the standard library
only and exits when the parent closes its stdin.
"""

import asyncio
import contextlib
import ssl
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, Optional
from urllib.parse import urlsplit, urlunsplit

RTSPS_DEFAULT_PORT = 322
COPY_CHUNK_BYTES = 65536
EXIT_TIMEOUT_S = 5


@dataclass
class Tunnel:
    url: str = ""
    # Why the tunnel could not reach the relay securely, once the tunnel has closed.
    error: Optional[str] = None


@contextlib.contextmanager
def verified_rtsps(url: str) -> Iterator[Tunnel]:
    """Yield a tunnel whose `url` is the plain RTSP equivalent of the given `rtsps://` url."""
    parts = urlsplit(url)
    if parts.scheme != "rtsps" or not parts.hostname:
        raise ValueError("expected an rtsps:// url")
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            __file__,
            parts.hostname,
            str(parts.port or RTSPS_DEFAULT_PORT),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tunnel = Tunnel()
    try:
        local = f"127.0.0.1:{int(process.stdout.readline())}"
        credentials = parts.netloc.rpartition("@")[0]
        netloc = f"{credentials}@{local}" if credentials else local
        tunnel.url = urlunsplit(("rtsp", netloc, parts.path, parts.query, ""))
        yield tunnel
    finally:
        # Closes the child's stdin, which ends it.
        try:
            _, stderr = process.communicate(timeout=EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
        tunnel.error = stderr.strip() or None


def _serve(host: str, port: int) -> None:
    context = ssl.create_default_context()

    async def forward(local_reader, local_writer) -> None:
        try:
            remote_reader, remote_writer = await asyncio.open_connection(
                host, port, ssl=context, server_hostname=host
            )
        except (OSError, ssl.SSLError) as error:
            print(
                f"cannot reach {host}:{port} securely: {error}",
                file=sys.stderr,
                flush=True,
            )
            local_writer.close()
            return
        await asyncio.gather(
            _copy(local_reader, remote_writer),
            _copy(remote_reader, local_writer),
            return_exceptions=True,
        )

    async def main() -> None:
        server = await asyncio.start_server(forward, "127.0.0.1", 0)
        print(server.sockets[0].getsockname()[1], flush=True)
        # Serve until the parent closes stdin, including when the parent dies.
        await asyncio.get_running_loop().run_in_executor(None, sys.stdin.read)

    asyncio.run(main())


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(COPY_CHUNK_BYTES):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


if __name__ == "__main__":
    _serve(sys.argv[1], int(sys.argv[2]))
