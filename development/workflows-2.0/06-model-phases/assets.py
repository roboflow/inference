"""Pinned identities of the trained weights and the repository images.

Nothing here downloads. ``locate_weights`` only looks in the given directory
or the default torch hub checkpoint directory; ``prepare_assets.py`` is the
explicit, opt-in way to fetch the pinned file on another machine.
"""

import hashlib
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import torch
import torchvision
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData

REPOSITORY = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PinnedFile:
    """A file identified by name, size and SHA-256.

    Attributes:
        name: File name.
        size: Size in bytes.
        sha256: Full lowercase hex digest.
        source: Where the file comes from (URL or repository path).
    """

    name: str
    size: int
    sha256: str
    source: str


RESNET18_WEIGHTS = PinnedFile(
    name="resnet18-f37072fd.pth",
    size=46_830_571,
    sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    source="https://download.pytorch.org/models/resnet18-f37072fd.pth",
)
"""torchvision ``ResNet18_Weights.IMAGENET1K_V1`` (ImageNet-1K, 1000 classes)."""

IMAGES: Dict[str, PinnedFile] = {
    "dogs": PinnedFile(
        name="dogs.jpg",
        size=57_523,
        sha256="83bfa4e706f274ce1da7309cec6374d542f9938b3538481035588681cdaff139",
        source="workflows/tests/assets/dogs.jpg",
    ),
    "beagle": PinnedFile(
        name="dog.jpeg",
        size=43_665,
        sha256="6d82ee3ded7902be612d06f8a978dbe02277c36bc56203c3714efe20b0039b80",
        source="tests/google_colab/assets/dog.jpeg",
    ),
    "car": PinnedFile(
        name="car.jpg",
        size=89_581,
        sha256="2c3e00a72fa1a8c346289e7551890ffe59ae1909dd667cc94a5b16b943c80a27",
        source="tests/workflows/integration_tests/execution/assets/car.jpg",
    ),
}
"""Existing repository images, kept where they are."""


class AssetError(RuntimeError):
    """A pinned file is missing or does not match its recorded identity."""


def sha256_of(path: Path) -> str:
    """Hash a file in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    hex_digest = digest.hexdigest()

    return hex_digest


def verify(path: Path, pinned: PinnedFile) -> Path:
    """Check that a file exists and matches its pinned size and hash.

    Args:
        path: Candidate file.
        pinned: Expected identity.

    Returns:
        The same path.

    Raises:
        AssetError: When the file is missing or differs.
    """
    if not path.is_file():
        raise AssetError(f"{pinned.name} not found at {path}")

    size = path.stat().st_size
    if size != pinned.size:
        raise AssetError(f"{path} has {size} bytes, expected {pinned.size}")

    digest = sha256_of(path)
    if digest != pinned.sha256:
        raise AssetError(f"{path} has SHA-256 {digest}, expected {pinned.sha256}")

    return path


def default_weights_dir() -> Path:
    """Return torch's checkpoint directory without importing or calling torch hub.

    Returns:
        ``$TORCH_HOME/hub/checkpoints``, by default under ``~/.cache/torch``.
    """
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home is None:
        cache_home = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        torch_home = str(Path(cache_home) / "torch")

    directory = Path(torch_home) / "hub" / "checkpoints"

    return directory


def locate_weights(weights_dir: Optional[Path] = None) -> Path:
    """Find and verify the pinned ResNet-18 weights. Never downloads.

    Args:
        weights_dir: Directory holding ``resnet18-f37072fd.pth``; the torch
            hub checkpoint directory when omitted.

    Returns:
        Path of the verified file.

    Raises:
        AssetError: When the file is absent or differs; the message names the
            preparation command.
    """
    directory = default_weights_dir() if weights_dir is None else weights_dir
    path = directory / RESNET18_WEIGHTS.name
    try:
        verified = verify(path, RESNET18_WEIGHTS)
    except AssetError as error:
        raise AssetError(
            f"{error}. Run prepare_assets.py --download-to <dir> once, then pass "
            "--weights-dir <dir>."
        ) from error

    return verified


def describe_assets(weights_path: Path) -> Dict[str, Any]:
    """Record the identity of every external input of a demo run.

    Args:
        weights_path: The verified weights file that was loaded.

    Returns:
        Weights and image identities with full hashes, plus library versions.
    """
    record = {
        "weights": {**asdict(RESNET18_WEIGHTS), "path": str(weights_path)},
        "images": {name: asdict(pinned) for name, pinned in IMAGES.items()},
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
        },
    }

    return record


def load_image(name: str) -> ImageData:
    """Read one pinned repository image as an RGB CHW tensor image.

    Args:
        name: Key of ``IMAGES``.

    Returns:
        A workflow input image whose ``image_id`` is ``name``.

    Raises:
        AssetError: When the image differs from its pinned identity.
    """
    pinned = IMAGES[name]
    path = verify(REPOSITORY / pinned.source, pinned)
    bgr = cv2.imread(str(path))
    image = ImageData.from_numpy_rgb(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_id=name
    )

    return image
