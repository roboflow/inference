"""Install an exact retained wheel and exercise its public import/CLI surface."""

import json
import subprocess
import tempfile
from pathlib import Path

import click
from packaging.utils import canonicalize_name, parse_wheel_filename
from versioning import _final_version, _validate_inventory

IMPORTS = {
    "inference-sdk": "inference_sdk",
    "inference-cli": "inference_cli",
    "inference-models": "inference_models",
    "roboflow-workflows": "roboflow_workflows",
    "streamvision": "streamvision",
    "inference": "inference",
    "inference-core": "inference",
    "inference-cpu": "inference",
    "inference-gpu": "inference",
}


def _inventory(packages, *, context="package smoke"):
    raw_inventory = {item["name"]: item["version"] for item in packages}
    _validate_inventory(raw_inventory, context=context)
    inventory = {
        canonicalize_name(name): (
            _final_version(version) if canonicalize_name(name) in IMPORTS else version
        )
        for name, version in raw_inventory.items()
    }
    return inventory


def _smoke(wheel, *, python, state):
    name, version, _, _ = parse_wheel_filename(wheel.name)
    module = IMPORTS[name]
    key = f"{name}-py{python}"
    baseline = None
    if (state / "accepted-rc.json").exists():
        baseline = json.loads((state / "accepted-rc.json").read_text())["smoke"][key]
    with tempfile.TemporaryDirectory(prefix="release-smoke-") as temporary:
        environment = Path(temporary) / "venv"
        subprocess.run(["uv", "venv", "--python", python, str(environment)], check=True)
        executable = environment / "bin/python"
        constraints = []
        if baseline is not None:
            path = Path(temporary) / "constraints.txt"
            path.write_text(
                "\n".join(
                    f"{package}=={version}"
                    for package, version in baseline.items()
                    if package not in IMPORTS
                )
            )
            constraints = ["--constraint", str(path)]
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                *constraints,
                "--python",
                str(executable),
                "--prerelease",
                "allow",
                "--find-links",
                str(wheel.parent.resolve()),
                str(wheel.resolve()),
            ],
            check=True,
        )
        subprocess.run(["uv", "pip", "check", "--python", str(executable)], check=True)
        # cwd is outside the checkout, so imports must come from the installed wheel.
        command = (
            "import importlib, importlib.metadata as m; "
            f"assert m.version({str(name)!r}) == {str(version)!r}; "
            f"importlib.import_module({module!r})"
        )
        if name == "inference-sdk":
            command += "; from inference_sdk import InferenceHTTPClient; InferenceHTTPClient(api_url='http://localhost:9001')"
        subprocess.run([str(executable), "-c", command], cwd=temporary, check=True)
        if name == "inference-cli":
            subprocess.run(
                [str(environment / "bin/inference"), "--help"],
                cwd=temporary,
                check=True,
            )
        installed = subprocess.run(
            [
                str(executable),
                "-c",
                'import importlib.metadata as m,json; print(json.dumps([{"name":d.metadata["Name"],"version":d.version} for d in m.distributions()]))',
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=temporary,
        )
        inventory = _inventory(
            json.loads(installed.stdout), context=f"package smoke {key}"
        )
        if baseline is not None and inventory != baseline:
            raise ValueError(f"{key}: installed dependency graph differs from RC")
        destination = state / "smoke" / f"{key}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(inventory, sort_keys=True) + "\n")


@click.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--python",
    default="3.12",
)
@click.option(
    "--state",
    type=click.Path(path_type=Path),
    default=Path("release-state"),
)
def _main(directory, *, python, state):
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise click.ClickException(f"No wheels in {directory}")
    for wheel in wheels:
        _smoke(wheel, python=python, state=state)


if __name__ == "__main__":
    _main()
