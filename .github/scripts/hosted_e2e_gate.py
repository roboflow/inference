"""Authorize manual hosted tests on main or an approved release fast-track head."""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def _get(path: str):
    request = urllib.request.Request(
        f"https://api.github.com/repos/roboflow/inference/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    return result


def _authorize() -> tuple[str, str]:
    ref = os.environ["GITHUB_REF"]
    sha = os.environ["GITHUB_SHA"]
    if (
        os.environ.get("GITHUB_REPOSITORY") != "roboflow/inference"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("HOSTED_E2E_TARGET") not in {"staging", "production"}
        or re.fullmatch(r"[0-9a-f]{40}", sha) is None
    ):
        raise ValueError("invalid caller")

    if ref == os.environ["HOSTED_E2E_DEFAULT_REF"]:
        return ref, sha

    match = re.fullmatch(
        r"refs/heads/fast-track/post-v(\d+\.\d+\.\d+)-pt([1-9]\d*)", ref
    )
    if match is None:
        raise ValueError("not a release fast-track")

    branch = ref.removeprefix("refs/heads/")
    query = urllib.parse.urlencode(
        {"state": "open", "base": "main", "head": f"roboflow:{branch}", "per_page": 100}
    )
    prs = _get(f"pulls?{query}")
    approved = [
        pr
        for pr in prs
        if pr["head"]["repo"]["full_name"] == "roboflow/inference"
        and pr["head"]["ref"] == branch
        and pr["head"]["sha"] == sha
        and pr["base"]["ref"] == "main"
        and pr["state"] == "open"
        and not pr["draft"]
        and any(label["name"] == "fast-track-deploy-approved" for label in pr["labels"])
    ]
    if len(approved) != 1:
        raise ValueError("approved umbrella PR must match the dispatched SHA")

    # Use the release tag object, rather than a similarly named branch.
    tag = _get(f"git/ref/tags/v{match[1]}")["object"]
    if tag["type"] == "tag":
        tag = _get(f"git/tags/{tag['sha']}")["object"]
    if tag["type"] != "commit":
        raise ValueError("release must resolve to a commit")

    comparison = _get(f"compare/{tag['sha']}...{sha}")
    if (
        comparison["status"] not in {"ahead", "identical"}
        or comparison["merge_base_commit"]["sha"] != tag["sha"]
    ):
        raise ValueError("fast-track must descend from its release tag")

    return ref, sha


if __name__ == "__main__":
    try:
        approved_ref, approved_sha = _authorize()
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"approved_ref={approved_ref}\napproved_sha={approved_sha}\n")
        print("Trusted hosted E2E revision verified.")
    except Exception:
        print(
            "Hosted E2E requires a default-branch manual run or an approved "
            "release fast-track umbrella PR matching this commit.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
