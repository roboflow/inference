"""Confirm a manual same-repository dispatch and pin its commit for every job.

Any branch or tag may run the hosted suite, as before sharding. This check is
accident prevention only, not a credential boundary: the secrets are
repository-level and anyone with write access can dispatch the workflow.
"""

import os
import re
import sys
from pathlib import Path


def _authorize() -> tuple[str, str]:
    ref = os.environ["GITHUB_REF"]
    sha = os.environ["GITHUB_SHA"]
    if (
        os.environ.get("GITHUB_REPOSITORY") != "roboflow/inference"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("HOSTED_E2E_TARGET") not in {"staging", "production"}
        or re.fullmatch(r"[0-9a-f]{40}", sha) is None
        or not ref.startswith(("refs/heads/", "refs/tags/"))
    ):
        raise ValueError("invalid caller")

    return ref, sha


if __name__ == "__main__":
    try:
        approved_ref, approved_sha = _authorize()
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"approved_ref={approved_ref}\napproved_sha={approved_sha}\n")
        print("Manual same-repository hosted E2E dispatch verified.")
    except Exception:
        print(
            "Hosted E2E requires a manual same-repository dispatch of a branch "
            "or tag with a valid target.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
