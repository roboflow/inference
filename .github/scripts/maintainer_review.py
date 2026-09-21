"""Bridge completed Claude reviews and author escalations to a Slack thread.

Only the `notify` command receives a Slack credential. All network destinations
and notification routing are controlled here, never by an artifact or PR text.
Uses the Python standard library so the privileged job installs no PR packages.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

PASS = "😎 PR passes the vibe-check and trust-me-bro verification."
WORKFLOW = ".github/workflows/claude-pr-review.yml"
STATE_MARKER = "<!-- maintainer-review-slack:v1 "
BOT_ID = 41898282
APP_ID = 15368
SHA = re.compile(r"[0-9a-f]{40}")
COMMAND = re.compile(r"/maintainer-review[ \t]+(\S[\s\S]*)")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    def __init__(self, repo, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError("Invalid repository")
        self.repo = repo
        self.token = token

    def request(self, path, data=None, method=None):
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
        return urllib.request.build_opener(NoRedirect).open(request, timeout=30)

    def api(self, path, data=None, method=None):
        with self.request(path, data, method) as response:
            return json.load(response)

    def pages(self, path, key=None):
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
        return read_result(raw)


def read_result(raw):
    # Never extract files from a workflow artifact into the privileged checkout.
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist() != ["result.json"]:
            raise ValueError("Unexpected handoff artifact contents")
        if archive.getinfo("result.json").file_size > 8192:
            raise ValueError("Handoff result too large")
        return json.loads(archive.read("result.json"))


def actions_comment(comment):
    return (
        comment.get("user", {}).get("id") == BOT_ID
        and comment.get("user", {}).get("type") == "Bot"
        and (comment.get("performed_via_github_app") or {}).get("id") == APP_ID
    )


def eligible(pr, repo, automatic=False):
    return (
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


def fresh_pass(comments, run_id, attempt, head_sha):
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
    return max(passes, key=lambda c: c["id"], default=None)


def collect(github, number, head_sha, run_id, attempt):
    pr = github.api(f"pulls/{number}")
    if not eligible(pr, github.repo, automatic=True) or pr["head"]["sha"] != head_sha:
        return None
    passed = fresh_pass(
        github.pages(f"issues/{number}/comments"), run_id, attempt, head_sha
    )
    if not passed:
        return None
    return {
        "version": 1,
        "repository": github.repo,
        "pr_number": number,
        "head_sha": head_sha,
        "run_id": run_id,
        "run_attempt": attempt,
        "verdict": "pass",
        "comment_id": passed["id"],
    }


def automatic_handoff(github, event):
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
    artifacts = github.pages(f"actions/runs/{run_id}/artifacts", "artifacts")
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
    if not eligible(pr, github.repo, automatic=True):
        return None
    if pr["head"]["sha"] != result["head_sha"]:
        return None
    passed = fresh_pass(
        github.pages(f"issues/{number}/comments"), run_id, attempt, result["head_sha"]
    )
    if (
        not passed
        or passed["id"] != result["comment_id"]
        or not run["run_started_at"] <= passed["created_at"] <= run["updated_at"]
    ):
        return None
    return {
        "pr": pr,
        "key": f"pass:{result['head_sha']}",
        "kind": "Agent review passed — maintainer review requested",
        "comment_id": passed["id"],
        "reason": "",
    }


def escalation_handoff(github, event):
    if event["action"] != "created" or "pull_request" not in event["issue"]:
        return None
    original = event["comment"]
    match = COMMAND.fullmatch(original["body"].strip())
    if not match or original["user"]["type"] != "User":
        return None
    number = event["issue"]["number"]
    pr = github.api(f"pulls/{number}")
    if not eligible(pr, github.repo):
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
    return {
        "pr": pr,
        "key": f"escalation:{original['id']}",
        "kind": "Contributor escalation — maintainer decision requested",
        "comment_id": original["id"],
        "reason": match[1][:2000],
    }


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def state_signature(state, token):
    # The agent shares the github-actions bot identity, so author checks alone
    # cannot authenticate a saved Slack destination. Sign it with a key available
    # only to the notifier. This does not expose the token or grant Slack access.
    return hmac.new(
        token.encode(), canonical(state).encode(), hashlib.sha256
    ).hexdigest()


def load_state(comments, repo, number, channel, token):
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
                envelope["signature"], state_signature(state, token)
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
    return max(valid, key=lambda item: item[0], default=(None, None))


def save_state(github, state, comment_id, token):
    envelope = {"state": state, "signature": state_signature(state, token)}
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
        github.api(f"issues/comments/{comment_id}", {"body": body}, "PATCH")
    else:
        github.api(f"issues/{state['pr_number']}/comments", {"body": body})


def slack_post(token, payload):
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


def publish(github, handoff, channel, maintainers, token, post=slack_post):
    if not token or not re.fullmatch(r"C[A-Z0-9]+", channel):
        raise ValueError("Configure the Slack environment token and channel ID")
    ids = maintainers.split(",")
    if not ids or any(not re.fullmatch(r"[UW][A-Z0-9]+", item.strip()) for item in ids):
        raise ValueError("Configure comma-separated maintainer Slack user IDs")
    pr = handoff["pr"]
    number = pr["number"]
    comments = github.pages(f"issues/{number}/comments")
    comment_id, state = load_state(comments, github.repo, number, channel, token)
    if state and handoff["key"] in state["delivered"]:
        return False
    # Recheck immediately before posting: a push/close/draft transition can occur
    # while the notification waits in the concurrency queue.
    latest = github.api(f"pulls/{number}")
    automatic = handoff["key"].startswith("pass:")
    if (
        not eligible(latest, github.repo, automatic)
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
    posted = post(token, payload)
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
    save_state(github, state, comment_id, token)
    return True


def main():
    github = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    if sys.argv[1] == "collect":
        result = collect(
            github,
            int(os.environ["PR_NUMBER"]),
            os.environ["REVIEW_HEAD_SHA"],
            int(os.environ["GITHUB_RUN_ID"]),
            int(os.environ["GITHUB_RUN_ATTEMPT"]),
        )
        if result:
            Path(os.environ["RESULT_PATH"]).write_text(canonical(result))
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"ready={'true' if result else 'false'}\n")
    elif sys.argv[1] == "notify":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if event["repository"]["full_name"] != github.repo:
            raise ValueError("Repository mismatch")
        event_name = os.environ["GITHUB_EVENT_NAME"]
        if event_name == "workflow_run":
            handoff = automatic_handoff(github, event)
        elif event_name == "issue_comment":
            handoff = escalation_handoff(github, event)
        else:
            raise ValueError("Unsupported handoff event")
        if handoff:
            publish(
                github,
                handoff,
                os.environ["SLACK_REVIEW_CHANNEL_ID"],
                os.environ["SLACK_REVIEW_MAINTAINER_IDS"],
                os.environ["SLACK_REVIEW_BOT_TOKEN"],
            )
        else:
            print("No eligible maintainer handoff for this event.")
    else:
        raise ValueError("Expected collect or notify")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # HTTP exceptions can contain signed artifact URLs; never print request
        # objects or response bodies in this privileged workflow's logs.
        print(
            f"Handoff failed ({type(error).__name__}). Inspect configuration and rerun.",
            file=sys.stderr,
        )
        sys.exit(1)
