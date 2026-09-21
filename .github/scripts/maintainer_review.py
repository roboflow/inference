# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click==8.5.0",
# ]
# ///

"""Bridge completed Claude reviews and author escalations to a Slack thread.

Only the `notify` command receives a Slack credential. All network destinations
and notification routing are controlled here, never by an artifact or PR text.
Click supplies the CLI; the privileged job installs no PR packages.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

import click

PASS = "😎 PR passes the vibe-check and trust-me-bro verification."
WORKFLOW = ".github/workflows/claude-pr-review.yml"
STATE_MARKER = "<!-- maintainer-review-slack:v1 "
BOT_ID = 41898282
APP_ID = 15368
SHA = re.compile(r"[0-9a-f]{40}")
COMMAND = re.compile(r"/maintainer-review[ \t]+(\S[\s\S]*)")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from forwarding credentials through redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Refuse automatic redirects that could forward credentials.

        Args:
            req (urllib.request.Request): Original HTTP request.
            fp (BinaryIO): Response stream for the redirect.
            code (int): HTTP redirect status code.
            msg (str): HTTP response reason phrase.
            headers (Mapping): Redirect response headers.
            newurl (str): Proposed redirect destination; deliberately not followed.
        """
        return None


class GitHub:
    """Access one GitHub repository with bounded, explicit API operations.

    Args:
        repo (str): Repository in owner/name form.
        token (str): GitHub credential scoped to the required repository operations.
    """

    def __init__(self, repo, *, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError("Invalid repository")

        self.repo = repo
        self.token = token

    def request(self, path, *, data=None, method=None):
        """Open a GitHub API request without following redirects.

        Args:
            path (str): Repository-relative GitHub API endpoint.
            data (dict | None): JSON request body, or None for a read-only request.
            method (str | None): Explicit HTTP method, or None to infer GET or POST.

        Returns:
            HTTPResponse: Open response stream; the caller must close it.
        """
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repo}/{path}",
            data=json.dumps(data).encode() if data is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        # Do not forward Authorization across redirects (artifact downloads use
        # signed storage URLs). Also refuse unexpected redirects from JSON APIs.
        result = urllib.request.build_opener(NoRedirect).open(request, timeout=30)

        return result

    def api(self, path, *, data=None, method=None):
        """Send a GitHub API request and decode its JSON response.

        Args:
            path (str): Repository-relative GitHub API endpoint.
            data (dict | None): JSON request body, or None for a read-only request.
            method (str | None): Explicit HTTP method, or None to infer GET or POST.

        Returns:
            dict | list: Decoded GitHub response.
        """
        with self.request(path, data=data, method=method) as response:
            result = json.load(response)

            return result

    def pages(self, path, *, key=None):
        """Read every page of a GitHub collection.

        Args:
            path (str): Repository-relative GitHub API endpoint.
            key (str | None): Response field containing page items; None for a top-level list.

        Returns:
            list[dict]: Combined collection items.
        """
        result = []
        for page in range(1, 101):
            separator = "&" if "?" in path else "?"
            value = self.api(f"{path}{separator}per_page=100&page={page}")
            items = value[key] if key else value
            result.extend(items)
            if len(items) < 100:
                return result
        raise ValueError("Pagination limit reached; refusing incomplete history")

    def artifact(self, artifact_id):
        """Read a bounded review artifact without forwarding GitHub credentials.

        Args:
            artifact_id (int): ID of the review result artifact to download.

        Returns:
            dict: Parsed review result.
        """
        try:
            response = self.request(f"actions/artifacts/{artifact_id}/zip")
        except urllib.error.HTTPError as error:
            if error.code != 302:
                raise

            location = error.headers["Location"]
            parsed = urllib.parse.urlsplit(location)
            if parsed.scheme != "https" or parsed.username or parsed.password:
                raise ValueError("Invalid artifact redirect") from None

            # Deliberately no GitHub Authorization header on this request.
            response = urllib.request.urlopen(location, timeout=30)
        with response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("Handoff artifact too large")

        result = read_result(raw)

        return result


def read_result(raw):
    """Validate and parse the review result without extracting archive files.

    Args:
        raw (bytes): Bounded ZIP archive containing only result.json.

    Returns:
        dict: Parsed handoff result, pending semantic validation.

    Raises:
        ValueError: If archive contents or size violate the handoff contract.
        zipfile.BadZipFile: If the downloaded data is not a valid ZIP archive.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist() != ["result.json"]:
            raise ValueError("Unexpected handoff artifact contents")

        if archive.getinfo("result.json").file_size > 8192:
            raise ValueError("Handoff result too large")

        result = json.loads(archive.read("result.json"))

        return result


def actions_comment(comment):
    """Identify comments posted through the GitHub Actions app.

    Args:
        comment (dict): GitHub issue comment including author and app metadata.

    Returns:
        bool: Whether both the bot identity and app identity match.
    """
    is_actions_comment = (
        comment.get("user", {}).get("id") == BOT_ID
        and comment.get("user", {}).get("type") == "Bot"
        and (comment.get("performed_via_github_app") or {}).get("id") == APP_ID
    )

    return is_actions_comment


def eligible(pr, *, repo, automatic=False):
    """Check whether a PR can enter the maintainer handoff.

    Args:
        pr (dict): Current pull request metadata from GitHub.
        repo (str): Repository in owner/name form.
        automatic (bool): Whether to apply agent-review scope and opt-out restrictions.

    Returns:
        bool: Whether the current PR state meets the handoff rules.
    """
    is_eligible = (
        pr["state"] == "open"
        and not pr["draft"]
        and pr["base"]["ref"] == "main"
        and pr["base"]["repo"]["full_name"] == repo
        and (
            not automatic
            or (
                pr["head"]["repo"]["full_name"] == repo
                and not any(
                    label["name"].lower() == "skip-claude-review"
                    for label in pr["labels"]
                )
            )
        )
    )

    return is_eligible


def fresh_pass(comments, *, run_id, attempt, head_sha):
    """Find an unedited sign-off from the latest review attempt.

    Args:
        comments (list[dict]): Complete paginated PR issue-comment history.
        run_id (int): ID of the workflow run that performed the review.
        attempt (int): Attempt number within the review workflow run.
        head_sha (str): Exact PR revision that the agent reviewed.

    Returns:
        dict | None: Matching pass comment, or None when superseded or absent.
    """
    marker = f"<!-- claude-pr-review:run={run_id}:{attempt} -->"
    starts = [
        c
        for c in comments
        if actions_comment(c)
        and marker in c["body"].splitlines()
        and f"**Claude review started** at commit `{head_sha}`." in c["body"]
    ]
    if len(starts) != 1:
        return None

    start = starts[0]
    if any(
        actions_comment(c)
        and c["id"] > start["id"]
        and "<!-- claude-pr-review:run-marker -->" in c["body"].splitlines()
        for c in comments
    ):
        return None  # A later review supersedes this result, even at the same SHA.

    passes = [
        c
        for c in comments
        if actions_comment(c)
        and c["body"] == PASS
        and c["id"] > start["id"]
        and c["created_at"] >= start["created_at"]
        and c["updated_at"] == c["created_at"]
    ]
    result = max(passes, key=lambda c: c["id"], default=None)

    return result


def collect(github, *, number, head_sha, run_id, attempt):
    """Record a fresh agent pass for the exact reviewed revision.

    Args:
        github (GitHub): Repository-scoped GitHub client.
        number (int): PR number within the configured repository.
        head_sha (str): Exact PR revision that the agent reviewed.
        run_id (int): ID of the workflow run that performed the review.
        attempt (int): Attempt number within the review workflow run.

    Returns:
        dict | None: Handoff result, or None when no current pass is available.
    """
    pr = github.api(f"pulls/{number}")
    if (
        not eligible(pr, repo=github.repo, automatic=True)
        or pr["head"]["sha"] != head_sha
    ):
        return None

    passed = fresh_pass(
        github.pages(f"issues/{number}/comments"),
        run_id=run_id,
        attempt=attempt,
        head_sha=head_sha,
    )
    if not passed:
        return None

    review_result = {
        "version": 1,
        "repository": github.repo,
        "pr_number": number,
        "head_sha": head_sha,
        "run_id": run_id,
        "run_attempt": attempt,
        "verdict": "pass",
        "comment_id": passed["id"],
    }

    return review_result


def automatic_handoff(github, *, event):
    """Validate a completed review before requesting maintainer attention.

    Args:
        github (GitHub): Repository-scoped GitHub client.
        event (dict): Original GitHub webhook payload for the triggering event.

    Returns:
        dict | None: Validated handoff, or None for an ineligible event.
    """
    trigger = event["workflow_run"]
    run_id, attempt = trigger["id"], trigger["run_attempt"]
    run = github.api(f"actions/runs/{run_id}/attempts/{attempt}")
    if (
        run["path"] != WORKFLOW
        or run["name"] != "Claude PR Review"
        or run["event"] not in {"pull_request", "workflow_dispatch"}
        or run["head_repository"]["full_name"] != github.repo
        or run["status"] != "completed"
        or run["conclusion"] != "success"
        or run["run_attempt"] != attempt
    ):
        return None

    artifacts = github.pages(f"actions/runs/{run_id}/artifacts", key="artifacts")
    matches = [
        a
        for a in artifacts
        if a["name"] == f"maintainer-review-result-{attempt}" and not a["expired"]
    ]
    if not matches:
        return None  # Blocked, skipped, old workflow version, or expired result.

    if len(matches) != 1:
        raise ValueError("Ambiguous handoff artifacts")

    result = github.artifact(matches[0]["id"])
    keys = {
        "version",
        "repository",
        "pr_number",
        "head_sha",
        "run_id",
        "run_attempt",
        "verdict",
        "comment_id",
    }
    if (
        not isinstance(result, dict)
        or set(result) != keys
        or result["version"] != 1
        or result["repository"] != github.repo
        or result["run_id"] != run_id
        or result["run_attempt"] != attempt
        or result["verdict"] != "pass"
        or type(result["pr_number"]) is not int
        or result["pr_number"] <= 0
        or type(result["comment_id"]) is not int
        or not isinstance(result["head_sha"], str)
        or not SHA.fullmatch(result["head_sha"])
    ):
        raise ValueError("Invalid handoff result")

    number = result["pr_number"]
    pr = github.api(f"pulls/{number}")
    if not eligible(pr, repo=github.repo, automatic=True):
        return None

    if pr["head"]["sha"] != result["head_sha"]:
        return None

    passed = fresh_pass(
        github.pages(f"issues/{number}/comments"),
        run_id=run_id,
        attempt=attempt,
        head_sha=result["head_sha"],
    )
    if (
        not passed
        or passed["id"] != result["comment_id"]
        or not run["run_started_at"] <= passed["created_at"] <= run["updated_at"]
    ):
        return None

    handoff = {
        "pr": pr,
        "key": f"pass:{result['head_sha']}",
        "kind": "Agent review passed — maintainer review requested",
        "comment_id": passed["id"],
        "reason": "",
    }

    return handoff


def escalation_handoff(github, *, event):
    """Authorize a contributor or maintainer escalation request.

    Args:
        github (GitHub): Repository-scoped GitHub client.
        event (dict): Original GitHub webhook payload for the triggering event.

    Returns:
        dict | None: Validated escalation, or None for an ineligible request.
    """
    if event["action"] != "created" or "pull_request" not in event["issue"]:
        return None

    original = event["comment"]
    match = COMMAND.fullmatch(original["body"].strip())
    if not match or original["user"]["type"] != "User":
        return None

    number = event["issue"]["number"]
    pr = github.api(f"pulls/{number}")
    if not eligible(pr, repo=github.repo):
        return None

    current = github.api(f"issues/comments/{original['id']}")
    if (
        current["body"] != original["body"]
        or current["user"]["id"] != original["user"]["id"]
    ):
        return None  # Deleted/edited requests cannot be replayed as new requests.

    if current["user"]["id"] != pr["user"]["id"]:
        login = urllib.parse.quote(current["user"]["login"], safe="")
        try:
            permission = github.api(f"collaborators/{login}/permission")["permission"]
        except urllib.error.HTTPError as error:
            if error.code not in {403, 404}:
                raise

            return None
        if permission not in {"write", "maintain", "admin"}:
            return None
    handoff = {
        "pr": pr,
        "key": f"escalation:{original['id']}",
        "kind": "Contributor escalation — maintainer decision requested",
        "comment_id": original["id"],
        "reason": match[1][:2000],
    }

    return handoff


def canonical(value):
    """Encode JSON consistently for signatures and saved state.

    Args:
        value (object): JSON-serializable data to encode deterministically.

    Returns:
        str: Deterministic compact JSON.
    """
    result = json.dumps(value, sort_keys=True, separators=(",", ":"))

    return result


def state_signature(state, *, token):
    """Authenticate saved Slack routing with the notifier-only credential.

    Args:
        state (dict): PR-bound Slack thread routing and delivered-event history.
        token (str): Credential for this client or notification step; never logged.

    Returns:
        str: Hexadecimal HMAC signature bound to the complete state.
    """
    # The agent shares the github-actions bot identity, so author checks alone
    # cannot authenticate a saved Slack destination. Sign it with a key available
    # only to the notifier. This does not expose the token or grant Slack access.
    result = hmac.new(
        token.encode(), canonical(state).encode(), hashlib.sha256
    ).hexdigest()

    return result


def load_state(comments, *, repo, number, channel, token):
    """Find the latest authentic Slack thread state for this PR.

    Args:
        comments (list[dict]): Complete paginated PR issue-comment history.
        repo (str): Repository in owner/name form.
        number (int): PR number within the configured repository.
        channel (str): Configured Slack channel ID.
        token (str): Credential for this client or notification step; never logged.

    Returns:
        tuple[int | None, dict | None]: State comment ID and state, or two None values.
    """
    valid = []
    for comment in comments:
        body = comment["body"]
        if not actions_comment(comment) or STATE_MARKER not in body:
            continue

        try:
            encoded = body.split(STATE_MARKER, 1)[1].split(" -->", 1)[0]
            envelope = json.loads(encoded)
            state = envelope["state"]
            if not hmac.compare_digest(
                envelope["signature"], state_signature(state, token=token)
            ):
                continue

            if (
                state["repository"] == repo
                and state["pr_number"] == number
                and state["channel"] == channel
                and re.fullmatch(r"\d+\.\d+", state["thread_ts"])
            ):
                valid.append((comment["id"], state))
        except (ValueError, KeyError, TypeError):
            continue
    result = max(valid, key=lambda item: item[0], default=(None, None))

    return result


def save_state(github, *, state, comment_id, token):
    """Create or update the PR comment containing signed Slack thread state.

    Args:
        github (GitHub): Repository-scoped GitHub client.
        state (dict): PR-bound Slack thread routing and delivered-event history.
        comment_id (int | None): Existing state-comment ID, or None to create one.
        token (str): Credential for this client or notification step; never logged.
    """
    envelope = {"state": state, "signature": state_signature(state, token=token)}
    # This integration is scoped to roboflow/inference and its Slack workspace.
    permalink = f"https://roboflow.slack.com/archives/{state['channel']}/p{state['thread_ts'].replace('.', '')}"
    body = (
        f"Maintainer review discussion: [Slack thread]({permalink}).\n\n"
        "Final approval and merge remain in GitHub.\n\n"
        f"{STATE_MARKER}{canonical(envelope)} -->"
    )
    if len(body) > 60000:
        raise ValueError("Handoff history exceeds GitHub comment capacity")

    if comment_id:
        github.api(f"issues/comments/{comment_id}", data={"body": body}, method="PATCH")
    else:
        github.api(f"issues/{state['pr_number']}/comments", data={"body": body})


def slack_post(*, token, payload):
    """Send one Slack message without automatically retrying ambiguous writes.

    Args:
        token (str): Credential for this client or notification step; never logged.
        payload (dict): Slack message fields prepared by the notifier.

    Returns:
        dict: Successful Slack response containing the channel and message timestamp.
    """
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    # Do not automatically retry POST: Slack may have accepted a timed-out call.
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise ValueError(
            f"Slack rejected handoff: {result.get('error', 'unknown_error')}"
        )

    return result


def publish(github, *, handoff, channel, maintainers, token, post=slack_post):
    """Post a current handoff and persist its deduplication state.

    Args:
        github (GitHub): Repository-scoped GitHub client.
        handoff (dict): Validated review or escalation with its PR and event key.
        channel (str): Configured Slack channel ID.
        maintainers (str): Comma-separated Slack user IDs to mention.
        token (str): Credential for this client or notification step; never logged.
        post (Callable): Message sender accepting token and payload keyword arguments.

    Returns:
        bool: True after delivery, or False for a duplicate or stale handoff.

    Raises:
        ValueError: If Slack configuration or its response is invalid.
    """
    if not token or not re.fullmatch(r"C[A-Z0-9]+", channel):
        raise ValueError("Configure the Slack environment token and channel ID")

    ids = maintainers.split(",")
    if not ids or any(not re.fullmatch(r"[UW][A-Z0-9]+", item.strip()) for item in ids):
        raise ValueError("Configure comma-separated maintainer Slack user IDs")

    pr = handoff["pr"]
    number = pr["number"]
    comments = github.pages(f"issues/{number}/comments")
    comment_id, state = load_state(
        comments, repo=github.repo, number=number, channel=channel, token=token
    )
    if state and handoff["key"] in state["delivered"]:
        return False

    # Recheck immediately before posting: a push/close/draft transition can occur
    # while the notification waits in the concurrency queue.
    latest = github.api(f"pulls/{number}")
    automatic = handoff["key"].startswith("pass:")
    if (
        not eligible(latest, repo=github.repo, automatic=automatic)
        or latest["head"]["sha"] != pr["head"]["sha"]
    ):
        return False

    url = f"https://github.com/{github.repo}/pull/{number}"
    mentions = " ".join(f"<@{item.strip()}>" for item in ids)
    payload = {
        "channel": channel,
        "text": f"Maintainer review requested for {github.repo} PR #{number}",
        "unfurl_links": False,
        "unfurl_media": False,
        "client_msg_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}:{handoff['key']}")),
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{mentions}\n*{handoff['kind']}*"},
            },
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": f"#{number}: {pr['title'][:500]}\nAuthor: {pr['user']['login']}\nRevision: {pr['head']['sha']}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"<{url}|Open PR> · <{url}#issuecomment-{handoff['comment_id']}|Review / escalation>\nReply in this thread to coordinate. Record final approval in GitHub.",
                },
            },
        ],
    }
    if handoff["reason"]:
        # Plain text prevents contributor-controlled @channel/user mentions.
        payload["blocks"].append(
            {
                "type": "section",
                "text": {"type": "plain_text", "text": handoff["reason"]},
            }
        )
    if state:
        payload["thread_ts"] = state["thread_ts"]
    posted = post(token=token, payload=payload)
    if posted["channel"] != channel or not re.fullmatch(r"\d+\.\d+", posted["ts"]):
        raise ValueError("Unexpected Slack message destination")

    if state is None:
        state = {
            "repository": github.repo,
            "pr_number": number,
            "channel": channel,
            "thread_ts": posted["ts"],
            "delivered": [],
        }
    state["delivered"].append(handoff["key"])
    print(f"Posted handoff for PR #{number}: channel={channel} ts={posted['ts']}")
    save_state(github, state=state, comment_id=comment_id, token=token)
    return True


@click.group()
def main():
    """Collect or deliver maintainer review handoffs from GitHub Actions."""


@main.command(
    name="collect",
)
def collect_command():
    """Record the review result using the collector's GitHub environment.

    Raises:
        click.ClickException: If collection or result persistence fails.
    """
    _execute(_collect_from_environment)


@main.command(
    name="notify",
)
def notify_command():
    """Deliver an eligible event using the notifier's protected environment.

    Raises:
        click.ClickException: If validation, delivery, or persistence fails.
    """
    _execute(_notify_from_environment)


def _execute(operation):
    """Report failures without exposing credentials or signed download URLs."""
    try:
        operation()
    except Exception as error:
        raise click.ClickException(
            f"Handoff failed ({type(error).__name__}). Inspect configuration and rerun."
        ) from None


def _collect_from_environment():
    github = GitHub(os.environ["GITHUB_REPOSITORY"], token=os.environ["GH_TOKEN"])
    result = collect(
        github,
        number=int(os.environ["PR_NUMBER"]),
        head_sha=os.environ["REVIEW_HEAD_SHA"],
        run_id=int(os.environ["GITHUB_RUN_ID"]),
        attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
    )

    if result:
        result_path = Path(os.environ["RESULT_PATH"])
        result_path.write_text(canonical(result), encoding="utf-8")

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open(mode="a", encoding="utf-8") as output:
        output.write(f"ready={'true' if result else 'false'}\n")


def _notify_from_environment():
    github = GitHub(os.environ["GITHUB_REPOSITORY"], token=os.environ["GH_TOKEN"])
    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    if event["repository"]["full_name"] != github.repo:
        raise ValueError("Repository mismatch")

    event_name = os.environ["GITHUB_EVENT_NAME"]
    if event_name == "workflow_run":
        handoff = automatic_handoff(github, event=event)
    elif event_name == "issue_comment":
        handoff = escalation_handoff(github, event=event)
    else:
        raise ValueError("Unsupported handoff event")

    if not handoff:
        click.echo("No eligible maintainer handoff for this event.")
        return

    publish(
        github,
        handoff=handoff,
        channel=os.environ["SLACK_REVIEW_CHANNEL_ID"],
        maintainers=os.environ["SLACK_REVIEW_MAINTAINER_IDS"],
        token=os.environ["SLACK_REVIEW_BOT_TOKEN"],
    )


if __name__ == "__main__":
    main()
