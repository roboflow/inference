import pytest
from filelock import FileLock

from build_scripts.download_fonts import DEFAULT_TARGET_DIR, download_fonts


@pytest.fixture(scope="session")
def bundled_fonts() -> None:
    """Ensure the approved font assets are available for rendering tests.

    Delegates to the shared downloader used by Docker builds and
    `make download_fonts` - downloads are checksum-verified and skipped when
    assets are already present, so the fixture is a no-op (and network-free)
    on provisioned environments.
    """
    DEFAULT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DEFAULT_TARGET_DIR / ".download.lock"

    with FileLock(str(lock_path), timeout=300):
        exit_code = download_fonts(DEFAULT_TARGET_DIR, only=[])

    assert exit_code == 0, "Failed to provision approved font assets for tests"
