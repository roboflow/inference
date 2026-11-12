"""Start the server inside a built DMG with isolated caches and wait until ready."""

import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import click

SERVER_EXECUTABLE = (
    Path("Roboflow Inference.app") / "Contents" / "Resources" / "inference-app"
)


def _environment(directory: Path, *, port: int) -> dict[str, str]:
    # A fresh HOME keeps the platformdirs caches of run_inference.py away from
    # the runner's, and a system-only PATH keeps build tools out of reach.
    env = {
        "HOME": str(directory),
        "TMPDIR": str(directory),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "DISABLE_VERSION_CHECK": "True",
        "API_LOGGING_ENABLED": "False",
        "TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS": "1",
    }

    return env


def _wait_until_ready(
    process: subprocess.Popen,
    *,
    url: str,
    timeout: int,
) -> float:
    """Poll ``url`` until it answers 200 and return the seconds that took."""
    started = time.monotonic()
    # Never consult user/system HTTP proxies, even for loopback requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Frozen server exited: {process.returncode}")

        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    ready_seconds = time.monotonic() - started
                    return ready_seconds
        except OSError:
            pass

        time.sleep(0.5)

    raise RuntimeError(f"Frozen server did not become ready in {timeout} seconds")


def _run_server(
    executable: Path,
    *,
    directory: Path,
    port: int,
    timeout: int,
) -> float:
    """Return the readiness time of ``executable``; print its log on failure."""
    log = directory / "server.log"
    with log.open("w") as output:
        # Its own session lets the cleanup stop the stream manager child too.
        process = subprocess.Popen(
            [str(executable)],
            cwd=directory,
            env=_environment(directory, port=port),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ready_seconds = _wait_until_ready(
                process,
                url=f"http://127.0.0.1:{port}/readiness",
                timeout=timeout,
            )
        except Exception:
            output.flush()
            print(log.read_text(errors="replace")[-20000:])
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            process.wait(timeout=30)

    return ready_seconds


@click.command()
@click.option(
    "--dmg",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="DMG built by build.py.",
)
@click.option(
    "--timeout",
    type=click.IntRange(
        min=1,
    ),
    default=300,
    show_default=True,
    help="Seconds to wait for /readiness.",
)
def main(*, dmg: Path, timeout: int) -> None:
    """Mount a built DMG read-only, start its server and wait for ``/readiness``.

    Args:
        dmg: DMG built by build.py.
        timeout: Seconds to wait for the server to report ready.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    with tempfile.TemporaryDirectory(prefix="inference-dmg-smoke-") as directory:
        root = Path(directory)
        volume = root / "volume"
        volume.mkdir()
        subprocess.run(
            [
                "hdiutil",
                "attach",
                "-nobrowse",
                "-readonly",
                "-mountpoint",
                str(volume),
                str(dmg.resolve()),
            ],
            check=True,
        )
        try:
            ready_seconds = _run_server(
                volume / SERVER_EXECUTABLE,
                directory=root,
                port=port,
                timeout=timeout,
            )
        finally:
            subprocess.run(
                ["hdiutil", "detach", str(volume), "-force"],
                check=False,
            )

    print(f"Server ready after {ready_seconds:.1f} seconds")


if __name__ == "__main__":
    main()
