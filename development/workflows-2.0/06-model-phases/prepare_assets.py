"""Check, or explicitly fetch, the pinned ResNet-18 weights and repository images.

Without ``--download-to`` this only verifies. The demos themselves never
download anything.
"""

import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import click
from assets import (
    IMAGES,
    REPOSITORY,
    RESNET18_WEIGHTS,
    AssetError,
    locate_weights,
    verify,
)


def _download(destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / RESNET18_WEIGHTS.name
    with tempfile.NamedTemporaryFile(dir=destination_dir, delete=False) as partial:
        partial_path = Path(partial.name)
        with urllib.request.urlopen(RESNET18_WEIGHTS.source) as response:
            while chunk := response.read(1 << 20):
                partial.write(chunk)
    try:
        verify(partial_path, RESNET18_WEIGHTS)
    except AssetError:
        partial_path.unlink()
        raise

    partial_path.replace(destination)

    return destination


@click.command()
@click.option(
    "--weights-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory to check; the torch hub checkpoint directory by default.",
)
@click.option(
    "--download-to",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Fetch the pinned weights into this directory when they are not there.",
)
def main(weights_dir: Optional[Path], download_to: Optional[Path]) -> None:
    """Verify the pinned assets, downloading the weights only when asked.

    Args:
        weights_dir: Directory to check for the weights.
        download_to: Explicit download destination; implies checking there.
    """
    for name, pinned in IMAGES.items():
        verify(REPOSITORY / pinned.source, pinned)
        click.echo(f"ok      image {name}: {pinned.source} sha256={pinned.sha256}")

    directory = download_to if download_to is not None else weights_dir
    try:
        path = locate_weights(directory)
    except AssetError as error:
        if download_to is None:
            click.echo(f"missing {error}")
            sys.exit(1)
        click.echo(f"fetch   {RESNET18_WEIGHTS.source} ({RESNET18_WEIGHTS.size} bytes)")
        path = _download(download_to)

    click.echo(f"ok      weights: {path} sha256={RESNET18_WEIGHTS.sha256}")


if __name__ == "__main__":
    main()
