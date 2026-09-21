"""Server-side parity: `inference.core.env` parsing of the MQTT blocks policy.

Retained here (not in `workflows/tests`) because it imports the server package.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ALLOW_USER_HOST = "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST"
WHITELISTED_HOSTS = "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS"
_REPO_ROOT = str(Path(__file__).resolve().parents[5])


def _import_env(extra_env: Dict[str, str]) -> Dict[str, Any]:
    # env.py computes the values at import time: read them in a fresh interpreter
    env = {**os.environ}
    env.pop(ALLOW_USER_HOST, None)
    env.pop(WHITELISTED_HOSTS, None)
    env.update(extra_env)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import json; from inference.core import env; "
        f"print('RESULT' + json.dumps([env.{ALLOW_USER_HOST}, "
        f"env.{WHITELISTED_HOSTS}]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    line = [l for l in result.stdout.splitlines() if l.startswith("RESULT")][-1]
    allow_user_provided, allowlist = json.loads(line[len("RESULT") :])
    return {"allow_user_provided": allow_user_provided, "allowlist": allowlist}


def test_env_defaults_allow_user_hosts_without_allowlist() -> None:
    values = _import_env({})

    assert values == {"allow_user_provided": True, "allowlist": None}


def test_env_parses_the_allowlist_in_order_without_empty_entries() -> None:
    values = _import_env(
        {
            ALLOW_USER_HOST: "False",
            WHITELISTED_HOSTS: " broker-2:1883, ,broker-1 ,broker-2:1883,",
        }
    )

    # trimmed, blank entries dropped, otherwise exactly what the operator wrote
    assert values == {
        "allow_user_provided": False,
        "allowlist": ["broker-2:1883", "broker-1", "broker-2:1883"],
    }
