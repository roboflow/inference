"""Install the final freezer dependencies with an explicit torch build pair."""

import subprocess
import sys
import tempfile
from pathlib import Path

import click
from packaging_checks import TORCH_BUILDS


def _install_torch_pair(python: str, flavor: str, *, index_url: str) -> list[str]:
    torch_version, vision_version, _ = TORCH_BUILDS[flavor]
    pins = [f"torch=={torch_version}", f"torchvision=={vision_version}"]
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", "--index-url", index_url] + pins,
        check=True,
    )
    return pins


@click.command()
@click.option(
    "--flavor",
    type=click.Choice(["cpu", "gpu"]),
    required=True,
)
def main(flavor: str) -> None:
    """Install local wheels with constrained torch dependencies.

    Args:
        flavor: Requested CPU or GPU installer flavor.
    """
    root = Path(__file__).resolve().parents[2]
    index = "cpu" if flavor == "cpu" else "cu126"
    # make create_wheels has already installed PyPI CPU dependencies. Explicit
    # local-version pins replace them even when the CUDA public version is older.
    pins = _install_torch_pair(
        sys.executable, flavor, index_url=f"https://download.pytorch.org/whl/{index}"
    )
    pip = [sys.executable, "-m", "pip"]
    package = "inference" if flavor == "cpu" else "inference_gpu"
    wheels = []
    for name in [
        "roboflow_workflows",
        "streamvision",
        "inference_sdk",
        "inference_models",
        package,
    ]:
        matches = list((root / "dist").glob(f"{name}-*.whl"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {name} wheel, found {matches}")

        wheels.append(str(matches[0]))
    wheels[-1] += "[sam,transformers,clip,http,yolo-world,grounding-dino]"
    with tempfile.TemporaryDirectory(prefix="inference-windows-pins-") as directory:
        constraints = Path(directory) / "torch.txt"
        constraints.write_text("\n".join(pins) + "\n")
        subprocess.run(
            pip
            + [
                "install",
                "--constraint",
                str(constraints),
                "--find-links",
                str(root / "dist"),
            ]
            + wheels,
            check=True,
        )
    subprocess.run(pip + ["check"], check=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("packaging_checks.py")),
            "--flavor",
            flavor,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
