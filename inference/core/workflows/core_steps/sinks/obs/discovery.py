"""Locate the obs-websocket password from a local OBS Studio installation.

OBS enables authentication by default and stores the password in its own config
file. When `inference` runs on the same machine as OBS, that file is the
authoritative source, so a Workflow does not need to carry the credential at all.

Only ever consulted for local hosts: reading this machine's config to talk to a
different machine's OBS would send the wrong password over the network.
"""

import json
import os
import platform
from pathlib import Path
from typing import List, NamedTuple, Optional

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


class DiscoveredPassword(NamedTuple):
    password: str
    source: Path
    auth_required: bool
    server_enabled: bool


def is_local_host(host: str) -> bool:
    return host.strip().lower() in LOCAL_HOSTS


def candidate_config_paths() -> List[Path]:
    """obs-websocket config locations, most likely first, per platform."""
    system = platform.system()
    home = Path.home()
    relative = Path("plugin_config/obs-websocket/config.json")
    if system == "Darwin":
        return [home / "Library/Application Support/obs-studio" / relative]
    if system == "Windows":
        app_data = os.environ.get("APPDATA")
        return [Path(app_data) / "obs-studio" / relative] if app_data else []
    return [
        home / ".config/obs-studio" / relative,
        # Flatpak installs redirect the whole config tree.
        home / ".var/app/com.obsproject.Studio/config/obs-studio" / relative,
    ]


def discover_password() -> Optional[DiscoveredPassword]:
    """Read the local OBS websocket credentials, or None when unavailable.

    Never raises: a missing, unreadable or malformed config simply means the caller
    falls back to whatever password the Workflow supplied.
    """
    for path in candidate_config_paths():
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        password = config.get("server_password")
        if not isinstance(password, str):
            continue
        return DiscoveredPassword(
            password=password,
            source=path,
            auth_required=bool(config.get("auth_required", True)),
            server_enabled=bool(config.get("server_enabled", False)),
        )
    return None
