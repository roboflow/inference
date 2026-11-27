"""Install the retained SDK wheel and import it outside the source checkout."""

import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def smoke(directory):
    """Install and import one retained SDK wheel without changing its bytes.

    Args:
        directory: Directory containing exactly one inference-sdk wheel.

    Raises:
        ValueError: The wheel inventory, metadata, or final digest is invalid.
        subprocess.CalledProcessError: Environment setup, installation, or import fails.
    """
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("SDK smoke requires exactly one retained wheel")
    wheel = wheels[0].resolve()
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            p for p in archive.namelist() if p.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Ambiguous SDK metadata")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
    if metadata["Name"] != "inference-sdk":
        raise ValueError("Expected inference-sdk wheel")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run([sys.executable, "-m", "venv", temporary], check=True)
        python = str(Path(temporary) / "bin/python")
        subprocess.run(
            [python, "-m", "pip", "install", "--no-cache-dir", str(wheel)],
            cwd=temporary,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                python,
                "-I",
                "-c",
                "from pathlib import Path; import sys; "
                "import inference_sdk; from inference_sdk import InferenceHTTPClient; "
                "from importlib.metadata import version; "
                "assert Path(inference_sdk.__file__).is_relative_to(sys.prefix); "
                "assert version('inference-sdk') == sys.argv[1]; "
                "print(version('inference-sdk'), inference_sdk.__file__)",
                metadata["Version"],
            ],
            cwd=temporary,
            env=env,
            check=True,
        )
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != digest:
        raise ValueError("SDK wheel changed during smoke")
    print(f"Tested retained SDK wheel: {wheel.name} sha256={digest}")


if __name__ == "__main__":
    smoke(Path("/work/dist/sdk"))
