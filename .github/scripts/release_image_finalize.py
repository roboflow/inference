"""Finalize an accepted RC image without resolving or downloading dependencies."""

import json
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path


def _inventory():
    return {
        re.sub(r"[-_.]+", "-", item.metadata["Name"]).lower(): item.version
        for item in distributions()
    }


def _main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    before = _inventory()
    wheels = [
        str(root / "wheels" / item["file"])
        for name, item in manifest.items()
        if name in before and before[name] != item["version"]
    ]
    if wheels:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(root / "vendor") + os.pathsep + environment.get("PYTHONPATH", "")
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                "--no-cache-dir",
                "--no-compile",
                "--disable-pip-version-check",
                "--break-system-packages",
                *wheels,
            ],
            env=environment,
            check=True,
        )
    # Some server images import checkout sources ahead of site-packages.
    for source in (root / "sources").iterdir():
        target = Path.cwd() / source.name
        if (target / "__init__.py").is_file():
            shutil.copytree(source, target, dirs_exist_ok=True)
    expected = {
        name: manifest[name]["version"] if name in manifest else version
        for name, version in before.items()
    }
    if _inventory() != expected:
        raise ValueError("Finalization changed the tested dependency inventory")


if __name__ == "__main__":
    _main()
