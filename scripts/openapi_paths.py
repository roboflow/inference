"""Route parity tooling between the legacy HTTP server and the new server.

``dump`` writes the (path, method) pairs an app exposes with every
route-enabling flag switched on. ``compare`` fails when the legacy dump holds a
pair that neither the server dump nor the allow-list holds.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple
from unittest.mock import MagicMock

import click

REPO_ROOT = Path(__file__).resolve().parent.parent

IGNORED_METHODS = {"OPTIONS"}

LEGACY_ENVIRONMENT = {
    "LAMBDA": "False",
    "GCP_SERVERLESS": "False",
    "OFFLINE_MODE": "False",
    "GET_MODEL_REGISTRY_ENABLED": "True",
    "CORE_MODELS_ENABLED": "True",
    "CORE_MODEL_CLIP_ENABLED": "True",
    "CORE_MODEL_PE_ENABLED": "True",
    "CORE_MODEL_SAM_ENABLED": "True",
    "CORE_MODEL_SAM2_ENABLED": "True",
    "CORE_MODEL_SAM3_ENABLED": "True",
    "CORE_MODEL_OWLV2_ENABLED": "True",
    "CORE_MODEL_GAZE_ENABLED": "True",
    "CORE_MODEL_DOCTR_ENABLED": "True",
    "CORE_MODEL_EASYOCR_ENABLED": "True",
    "CORE_MODEL_TROCR_ENABLED": "True",
    "CORE_MODEL_PPOCR_ENABLED": "True",
    "CORE_MODEL_GROUNDINGDINO_ENABLED": "True",
    "CORE_MODEL_YOLO_WORLD_ENABLED": "True",
    "LMM_ENABLED": "True",
    "MOONDREAM2_ENABLED": "True",
    "DEPTH_ESTIMATION_ENABLED": "True",
    "SAM3_3D_OBJECTS_ENABLED": "True",
    "ACTION_RECOGNITION_ENABLED": "True",
    "LEGACY_ROUTE_ENABLED": "True",
    "DISABLE_WORKFLOW_ENDPOINTS": "False",
    "DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS": "False",
    "WEBRTC_WORKER_ENABLED": "True",
    "ENABLE_STREAM_API": "True",
    "ENABLE_BUILDER": "True",
    "ENABLE_DASHBOARD": "False",
    "SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED": "True",
}

SERVER_ENVIRONMENT = {
    "LAMBDA": "False",
    "GCP_SERVERLESS": "False",
    "OFFLINE_MODE": "False",
    "LEGACY_ROUTES_ENABLED": "True",
    "LEGACY_ROUTE_ENABLED": "True",
    "LEGACY_CONTROL_PLANE_ROUTES_ENABLED": "True",
    "GET_MODEL_REGISTRY_ENABLED": "True",
    "CORE_MODELS_ENABLED": "True",
    "CORE_MODEL_CLIP_ENABLED": "True",
    "CORE_MODEL_PE_ENABLED": "True",
    "CORE_MODEL_SAM_ENABLED": "True",
    "CORE_MODEL_SAM2_ENABLED": "True",
    "CORE_MODEL_SAM3_ENABLED": "True",
    "CORE_MODEL_OWLV2_ENABLED": "True",
    "CORE_MODEL_GAZE_ENABLED": "True",
    "CORE_MODEL_DOCTR_ENABLED": "True",
    "CORE_MODEL_EASYOCR_ENABLED": "True",
    "CORE_MODEL_TROCR_ENABLED": "True",
    "CORE_MODEL_PPOCR_ENABLED": "True",
    "CORE_MODEL_GROUNDINGDINO_ENABLED": "True",
    "CORE_MODEL_YOLO_WORLD_ENABLED": "True",
    "LMM_ENABLED": "True",
    "MOONDREAM2_ENABLED": "True",
    "DEPTH_ESTIMATION_ENABLED": "True",
    "SAM3_3D_OBJECTS_ENABLED": "True",
    "ACTION_RECOGNITION_ENABLED": "True",
    "DISABLE_WORKFLOW_ENDPOINTS": "False",
    "DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS": "False",
    "ENABLE_STREAM_API": "True",
    "ENABLE_BUILDER": "True",
    "ENABLE_DASHBOARD": "False",
    "ENABLE_PROMETHEUS": "True",
    "SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED": "True",
}


def _load_legacy_app():
    os.chdir(REPO_ROOT)
    from inference.core.interfaces.http.http_api import HttpInterface

    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    interface = HttpInterface(model_manager=model_manager)

    return interface.app


def _load_server_app():
    from inference_server.app import app

    return app


APPS = {
    "legacy": (LEGACY_ENVIRONMENT, _load_legacy_app),
    "server": (SERVER_ENVIRONMENT, _load_server_app),
}


def _iter_routes(app) -> list:
    from fastapi import routing

    iter_route_contexts = getattr(routing, "iter_route_contexts", None)
    if iter_route_contexts is None:
        return list(app.routes)

    return list(iter_route_contexts(app.routes))


def _collect_paths(app) -> Dict[str, List[str]]:
    collected: Dict[str, Set[str]] = {}
    for path, operations in app.openapi().get("paths", {}).items():
        collected.setdefault(path, set()).update(
            method.upper() for method in operations
        )
    for route in _iter_routes(app):
        methods = getattr(route, "methods", None)
        path = getattr(route, "path_format", None)
        if not methods or path is None:
            continue

        route_methods = {method.upper() for method in methods} - IGNORED_METHODS
        if "GET" in route_methods:
            route_methods.discard("HEAD")
        collected.setdefault(path, set()).update(route_methods)
    paths = {path: sorted(methods) for path, methods in sorted(collected.items())}

    return paths


def _read_pairs(path: Path) -> Set[Tuple[str, str]]:
    document = json.loads(path.read_text())
    pairs = {
        (method, route)
        for route, methods in document["paths"].items()
        for method in methods
    }

    return pairs


def _read_allowlist(path: Path) -> Set[Tuple[str, str]]:
    entries: Set[Tuple[str, str]] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        method, route = line.split(maxsplit=1)
        entries.add((method, route))

    return entries


def _format(pairs: Set[Tuple[str, str]]) -> List[str]:
    lines = [f"{method} {route}" for method, route in sorted(pairs, key=_sort_key)]

    return lines


def _sort_key(pair: Tuple[str, str]) -> Tuple[str, str]:
    return pair[1], pair[0]


@click.group()
def cli() -> None:
    """Dump and compare the routes exposed by the legacy and the new server."""


@cli.command()
@click.option(
    "--app",
    "app_name",
    type=click.Choice(
        ["legacy", "server"],
    ),
    required=True,
)
@click.option(
    "--output",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
def dump(app_name: str, output: Path) -> None:
    """Write the routes of one app, with every route-enabling flag on."""
    environment, loader = APPS[app_name]
    os.environ.update(environment)
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["MODEL_CACHE_DIR"] = cache_dir
        app = loader()
        paths = _collect_paths(app)
    if not paths:
        raise click.ClickException(f"No paths collected from the {app_name} app.")
    output.write_text(json.dumps({"paths": paths}, indent=2, sort_keys=True) + "\n")


@cli.command()
@click.option(
    "--legacy",
    "legacy_file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--server",
    "server_file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--allowlist",
    "allowlist_file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
def compare(legacy_file: Path, server_file: Path, allowlist_file: Path) -> None:
    """Fail when a legacy route is neither on the server nor allow-listed."""
    legacy = _read_pairs(legacy_file)
    server = _read_pairs(server_file)
    allowlist = _read_allowlist(allowlist_file)

    for name, inventory in (("legacy", legacy), ("server", server)):
        if not inventory:
            raise click.ClickException(f"The {name} inventory is empty.")

    missing = legacy - server - allowlist
    stale = allowlist - legacy
    no_longer_needed = (allowlist & legacy) & server

    if no_longer_needed:
        click.echo("Allow-list entries the server now serves (remove them):")
        for line in _format(no_longer_needed):
            click.echo(f"  {line}")
    failed = False
    if missing:
        failed = True
        click.echo("Legacy routes missing from the server and the allow-list:")
        for line in _format(missing):
            click.echo(line)
    if stale:
        failed = True
        click.echo("Allow-list entries absent from the legacy routes:")
        for line in _format(stale):
            click.echo(line)
    if failed:
        sys.exit(1)
    click.echo("Route parity holds.")


if __name__ == "__main__":
    cli()
