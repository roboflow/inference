"""Network-free behavior tests for the privileged review handoff."""

import copy
import io
import json
import unittest
import urllib.error
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import maintainer_review as bridge
from click.testing import CliRunner

REPO = "roboflow/inference"
HEAD = "a" * 40
TIME = "2026-09-21T10:00:00Z"
LATER = "2026-09-21T10:01:00Z"


def comment(number, *, body, bot=True):
    """Build a GitHub comment fixture with realistic app metadata.

    Args:
        number (int): Comment ID within the test fixture.
        body (str): Comment text to include in the test event.
        bot (bool): Whether the fixture represents the GitHub Actions bot.

    Returns:
        dict: Comment fixture for review and escalation tests.
    """
    comment_fixture = {
        "id": number,
        "body": body,
        "created_at": TIME,
        "updated_at": TIME,
        "user": {
            "id": bridge.BOT_ID if bot else 42,
            "type": "Bot" if bot else "User",
            "login": "author",
        },
        "performed_via_github_app": {"id": bridge.APP_ID} if bot else None,
    }

    return comment_fixture


class FakeGitHub:
    """Emulate repository responses and persisted comments without network access."""

    repo = REPO

    def __init__(self):
        self.pr = {
            "number": 2989,
            "state": "open",
            "draft": False,
            "title": "Improve inference",
            "base": {"ref": "main", "repo": {"full_name": REPO}},
            "head": {"sha": HEAD, "repo": {"full_name": REPO}},
            "labels": [],
            "user": {"id": 42, "login": "author"},
        }
        self.comments = [
            comment(
                10,
                body=f"<!-- claude-pr-review:run-marker -->\n<!-- claude-pr-review:run=100:1 -->\n🤖 **Claude review started** at commit `{HEAD}`.",
            ),
            comment(11, body=bridge.PASS),
        ]
        self.run = {
            "id": 100,
            "run_attempt": 1,
            "path": bridge.WORKFLOW,
            "name": "Claude PR Review",
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "head_repository": {"full_name": REPO},
            "run_started_at": TIME,
            "updated_at": LATER,
        }
        self.result = {
            "version": 1,
            "repository": REPO,
            "pr_number": 2989,
            "head_sha": HEAD,
            "run_id": 100,
            "run_attempt": 1,
            "verdict": "pass",
            "comment_id": 11,
        }
        self.artifacts = [
            {"id": 200, "name": "maintainer-review-result-1", "expired": False}
        ]
        self.permission = "read"
        self.fail_save = False

    def api(self, path, *, data=None, method=None):
        """Emulate GitHub reads and state-comment writes for a test.

        Args:
            path (str): Repository-relative GitHub API endpoint.
            data (dict | None): JSON request body, or None for a read-only request.
            method (str | None): Explicit HTTP method, or None to infer GET or POST.

        Returns:
            dict | list: Decoded GitHub response.
        """
        if path == "pulls/2989":
            result = copy.deepcopy(self.pr)

            return result

        if path == "actions/runs/100/attempts/1":
            return self.run

        if path.startswith("collaborators/"):
            return {"permission": self.permission}

        if path.startswith("issues/comments/"):
            value = next(
                c for c in self.comments if c["id"] == int(path.rsplit("/", 1)[1])
            )
            if method == "PATCH":
                if self.fail_save:
                    raise RuntimeError("save failed")

                value["body"] = data["body"]
            return value

        if path == "issues/2989/comments" and data is not None:
            if self.fail_save:
                raise RuntimeError("save failed")

            value = comment(max(c["id"] for c in self.comments) + 1, body=data["body"])
            self.comments.append(value)
            return value

        raise AssertionError(f"Unexpected API request: {path}")

    def pages(self, path, *, key=None):
        """Return the complete in-memory comment or artifact collection.

        Args:
            path (str): Repository-relative GitHub API endpoint.
            key (str | None): Response field containing page items; None for a top-level list.

        Returns:
            list[dict]: Combined collection items.
        """
        if path == "issues/2989/comments":
            return self.comments

        if path == "actions/runs/100/artifacts":
            return self.artifacts

        raise AssertionError(path)

    def artifact(self, artifact_id):
        """Return the fixture result for the expected artifact ID.

        Args:
            artifact_id (int): ID of the review result artifact to download.

        Returns:
            dict: Parsed review result.
        """
        assert artifact_id == 200
        return self.result


class HandoffTests(unittest.TestCase):
    """Exercise authorization, revision checks, delivery, and state integrity."""

    def setUp(self):
        """Create independent GitHub fixtures for each handoff test."""
        self.github = FakeGitHub()
        self.event = {"workflow_run": {"id": 100, "run_attempt": 1}}

    def automatic(self):
        """Evaluate the automatic handoff against the current test fixture.

        Returns:
            dict | None: Handoff returned by the production validator.
        """
        result = bridge.automatic_handoff(self.github, event=self.event)

        return result

    def escalation(
        self,
        body="/maintainer-review I disagree with the backwards-compatibility finding.",
    ):
        """Add a human comment and build its original webhook event.

        Args:
            body (str): Comment text to include in the test event.

        Returns:
            dict: Escalation webhook payload independent of subsequent fixture edits.
        """
        request = comment(20, body=body, bot=False)
        self.github.comments.append(request)
        event = {
            "action": "created",
            "issue": {"number": 2989, "pull_request": {}},
            "comment": copy.deepcopy(request),
        }

        return event

    def test_collect_and_notify_current_pass(self):
        """Verify collect and notify current pass."""
        self.assertEqual(
            bridge.collect(
                self.github, number=2989, head_sha=HEAD, run_id=100, attempt=1
            ),
            self.github.result,
        )
        self.assertEqual(self.automatic()["key"], f"pass:{HEAD}")

    def test_no_pass_for_findings_skip_or_missing_prompt(self):
        """Verify no pass for findings skip or missing prompt."""
        for body in (
            "Blocking correctness issue",
            "[review skipped — non-substantive change]",
            "",
        ):
            with self.subTest(body=body):
                self.github.comments[1]["body"] = body
                self.assertIsNone(
                    bridge.collect(
                        self.github, number=2989, head_sha=HEAD, run_id=100, attempt=1
                    )
                )
                self.assertIsNone(self.automatic())

    def test_success_without_result_is_not_approval(self):
        """Verify success without result is not approval."""
        self.github.artifacts = []
        self.assertIsNone(self.automatic())

    def test_human_cannot_forge_pass(self):
        """Verify human cannot forge pass."""
        self.github.comments[1] = comment(11, body=bridge.PASS, bot=False)
        self.assertIsNone(self.automatic())

    def test_wrong_app_cannot_forge_pass(self):
        """Verify wrong app cannot forge pass."""
        self.github.comments[1]["performed_via_github_app"] = {"id": 123}
        self.assertIsNone(self.automatic())

    def test_old_pass_cannot_be_reused(self):
        """Verify old pass cannot be reused."""
        self.github.comments[1]["id"] = 9
        self.assertIsNone(self.automatic())

    def test_edited_pass_is_rejected(self):
        """Verify edited pass is rejected."""
        self.github.comments[1]["updated_at"] = LATER
        self.assertIsNone(self.automatic())

    def test_push_invalidates_approval(self):
        """Verify push invalidates approval."""
        self.github.pr["head"]["sha"] = "b" * 40
        self.assertIsNone(self.automatic())

    def test_new_review_supersedes_older_pass(self):
        """Verify new review supersedes older pass."""
        self.github.comments.append(
            comment(12, body="<!-- claude-pr-review:run-marker -->\nNew review")
        )
        self.assertIsNone(self.automatic())

    def test_ineligible_pr_is_not_announced(self):
        """Verify ineligible pr is not announced."""
        for field, value in (
            ("draft", True),
            ("state", "closed"),
            ("labels", [{"name": "SKIP-CLAUDE-REVIEW"}]),
        ):
            with self.subTest(field=field):
                self.github = FakeGitHub()
                self.github.pr[field] = value
                self.assertIsNone(self.automatic())

    def test_wrong_workflow_failure_or_fork_is_rejected(self):
        """Verify wrong workflow failure or fork is rejected."""
        for field, value in (
            ("path", ".github/workflows/fake.yml"),
            ("conclusion", "failure"),
            ("head_repository", {"full_name": "other/fork"}),
            ("event", "push"),
        ):
            with self.subTest(field=field):
                self.github = FakeGitHub()
                self.github.run[field] = value
                self.assertIsNone(self.automatic())

    def test_dispatch_result_does_not_depend_on_associated_pr_list(self):
        """Verify dispatch result does not depend on associated pr list."""
        self.github.run["event"] = "workflow_dispatch"
        self.github.run["pull_requests"] = []
        self.assertIsNotNone(self.automatic())

    def test_pass_outside_run_window_is_rejected(self):
        """Verify pass outside run window is rejected."""
        self.github.run["updated_at"] = "2026-09-20T00:00:00Z"
        self.assertIsNone(self.automatic())

    def test_wrong_attempt_repository_sha_or_type_is_rejected(self):
        """Verify wrong attempt repository sha or type is rejected."""
        for field, value in (
            ("run_attempt", 2),
            ("run_id", 101),
            ("repository", "other/repo"),
            ("head_sha", "$(bad)"),
            ("pr_number", "2989"),
            ("verdict", "blocked"),
        ):
            with self.subTest(field=field):
                self.github = FakeGitHub()
                self.github.result[field] = value
                with self.assertRaises(ValueError):
                    self.automatic()

    def test_author_can_escalate(self):
        """Verify author can escalate."""
        result = bridge.escalation_handoff(self.github, event=self.escalation())
        self.assertEqual(result["key"], "escalation:20")
        self.assertTrue(result["reason"].startswith("I disagree"))

    def test_author_can_escalate_fork_and_opt_out(self):
        """Verify author can escalate fork and opt out."""
        self.github.pr["head"]["repo"]["full_name"] = "author/fork"
        self.github.pr["labels"] = [{"name": "skip-claude-review"}]
        self.assertIsNotNone(
            bridge.escalation_handoff(self.github, event=self.escalation())
        )

    def test_empty_reason_or_embedded_command_is_ignored(self):
        """Verify empty reason or embedded command is ignored."""
        for body in (
            "/maintainer-review",
            "/maintainer-review   ",
            "Please /maintainer-review why",
            "> /maintainer-review why",
        ):
            with self.subTest(body=body):
                self.github = FakeGitHub()
                self.assertIsNone(
                    bridge.escalation_handoff(self.github, event=self.escalation(body))
                )

    def test_unrelated_user_cannot_escalate_but_maintainer_can(self):
        """Verify unrelated user cannot escalate but maintainer can."""
        event = self.escalation()
        self.github.pr["user"]["id"] = 999
        self.assertIsNone(bridge.escalation_handoff(self.github, event=event))
        self.github.permission = "write"
        self.assertIsNotNone(bridge.escalation_handoff(self.github, event=event))

    def test_deleted_or_edited_escalation_is_not_replayed(self):
        """Verify deleted or edited escalation is not replayed."""
        event = self.escalation()
        self.github.comments[-1]["body"] = "Never mind"
        self.assertIsNone(bridge.escalation_handoff(self.github, event=event))

    def test_bot_cannot_request_human_escalation(self):
        """Verify bot cannot request human escalation."""
        event = self.escalation()
        event["comment"]["user"]["type"] = "Bot"
        self.assertIsNone(bridge.escalation_handoff(self.github, event=event))

    def test_post_retry_is_noop_and_escalation_reuses_thread(self):
        """Verify post retry is noop and escalation reuses thread."""
        post = Mock(return_value={"channel": "C123", "ts": "12345.000001"})
        handoff = self.automatic()
        self.assertTrue(
            bridge.publish(
                self.github,
                handoff=handoff,
                channel="C123",
                maintainers="U123,U456",
                token="secret",
                post=post,
            )
        )
        self.assertFalse(
            bridge.publish(
                self.github,
                handoff=handoff,
                channel="C123",
                maintainers="U123,U456",
                token="secret",
                post=post,
            )
        )
        self.assertEqual(post.call_count, 1)
        escalation = bridge.escalation_handoff(self.github, event=self.escalation())
        bridge.publish(
            self.github,
            handoff=escalation,
            channel="C123",
            maintainers="U123",
            token="secret",
            post=post,
        )
        self.assertEqual(post.call_args.kwargs["payload"]["thread_ts"], "12345.000001")
        self.assertEqual(
            sum(bridge.STATE_MARKER in c["body"] for c in self.github.comments), 1
        )
        self.assertNotIn("secret", self.github.comments[2]["body"])
        self.assertIn(
            "https://roboflow.slack.com/archives/C123/p12345000001",
            self.github.comments[2]["body"],
        )

    def test_forged_state_cannot_redirect_message(self):
        """Verify forged state cannot redirect message."""
        state = {
            "repository": REPO,
            "pr_number": 2989,
            "channel": "C123",
            "thread_ts": "999.000001",
            "delivered": [],
        }
        forged = {
            "state": state,
            "signature": bridge.state_signature(state, token="wrong-key"),
        }
        self.github.comments.append(
            comment(15, body=bridge.STATE_MARKER + json.dumps(forged) + " -->")
        )
        post = Mock(return_value={"channel": "C123", "ts": "123.000001"})
        bridge.publish(
            self.github,
            handoff=self.automatic(),
            channel="C123",
            maintainers="U123",
            token="secret",
            post=post,
        )
        self.assertNotIn("thread_ts", post.call_args.kwargs["payload"])

    def test_untrusted_title_and_reason_are_plain_text(self):
        """Verify untrusted title and reason are plain text."""
        self.github.pr["title"] = "<!channel> $(curl attacker)"
        handoff = bridge.escalation_handoff(
            self.github, event=self.escalation("/maintainer-review <@U999> <!channel>")
        )
        post = Mock(return_value={"channel": "C123", "ts": "123.000001"})
        bridge.publish(
            self.github,
            handoff=handoff,
            channel="C123",
            maintainers="U123",
            token="secret",
            post=post,
        )
        blocks = post.call_args.kwargs["payload"]["blocks"]
        self.assertEqual(blocks[1]["text"]["type"], "plain_text")
        self.assertEqual(blocks[-1]["text"]["type"], "plain_text")

    def test_push_while_waiting_does_not_post(self):
        """Verify push while waiting does not post."""
        handoff = self.automatic()
        self.github.pr["head"]["sha"] = "b" * 40
        post = Mock()
        self.assertFalse(
            bridge.publish(
                self.github,
                handoff=handoff,
                channel="C123",
                maintainers="U123",
                token="secret",
                post=post,
            )
        )
        post.assert_not_called()

    def test_slack_failure_does_not_record_delivery(self):
        """Verify slack failure does not record delivery."""
        post = Mock(side_effect=ValueError("Slack failed"))
        with self.assertRaises(ValueError):
            bridge.publish(
                self.github,
                handoff=self.automatic(),
                channel="C123",
                maintainers="U123",
                token="secret",
                post=post,
            )
        self.assertFalse(
            any(bridge.STATE_MARKER in c["body"] for c in self.github.comments)
        )

    def test_save_failure_surfaces_instead_of_claiming_success(self):
        """Verify save failure surfaces instead of claiming success."""
        self.github.fail_save = True
        post = Mock(return_value={"channel": "C123", "ts": "123.000001"})
        with self.assertRaises(RuntimeError):
            bridge.publish(
                self.github,
                handoff=self.automatic(),
                channel="C123",
                maintainers="U123",
                token="secret",
                post=post,
            )

    def test_bad_configuration_fails_before_posting(self):
        """Verify bad configuration fails before posting."""
        for channel, maintainers, token in (
            ("#release", "U123", "secret"),
            ("C123", "@everyone", "secret"),
            ("C123", "U123", ""),
        ):
            post = Mock()
            with self.assertRaises(ValueError):
                bridge.publish(
                    self.github,
                    handoff=self.automatic(),
                    channel=channel,
                    maintainers=maintainers,
                    token=token,
                    post=post,
                )
            post.assert_not_called()


class ArtifactTests(unittest.TestCase):
    """Exercise bounded archive parsing and safe artifact redirects."""

    def archive(self, name, *, content):
        """Create a ZIP fixture in memory.

        Args:
            name (str): File name to include in the test ZIP archive.
            content (str): File contents to store in the test ZIP archive.

        Returns:
            bytes: ZIP archive bytes.
        """
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name, content)
        result = raw.getvalue()

        return result

    def test_json_is_read_without_extracting(self):
        """Verify json is read without extracting."""
        self.assertEqual(
            bridge.read_result(self.archive("result.json", content='{"version":1}')),
            {"version": 1},
        )

    def test_path_traversal_and_large_content_are_rejected(self):
        """Verify path traversal and large content are rejected."""
        for name, content in (("../result.json", "{}"), ("result.json", " " * 9000)):
            with self.assertRaises(ValueError):
                bridge.read_result(self.archive(name, content=content))

    def test_artifact_redirect_does_not_forward_github_token(self):
        """Verify artifact redirect does not forward github token."""
        github = bridge.GitHub(REPO, token="github-secret")
        github.request = Mock(
            side_effect=urllib.error.HTTPError(
                "unused",
                302,
                "redirect",
                {"Location": "https://storage.example/result"},
                None,
            )
        )
        response = io.BytesIO(self.archive("result.json", content="{}"))
        with patch("urllib.request.urlopen", return_value=response) as download:
            self.assertEqual(github.artifact(1), {})
            download.assert_called_once_with(
                "https://storage.example/result", timeout=30
            )


class CliTests(unittest.TestCase):
    """Exercise command parsing and credential-safe CLI errors."""

    def test_help_does_not_require_credentials(self):
        """Display both commands without reading the Actions environment."""
        result = CliRunner().invoke(bridge.main, args=["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("collect", result.output)
        self.assertIn("notify", result.output)

    def test_unknown_command_does_not_start_notification(self):
        """Reject invalid commands before running the notifier."""
        with patch.object(bridge, "_notify_from_environment") as notify:
            result = CliRunner().invoke(bridge.main, args=["unknown"])

        self.assertEqual(result.exit_code, 2)
        notify.assert_not_called()

    def test_collector_writes_result_and_action_output(self):
        """Persist a validated pass and report readiness to GitHub Actions."""
        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            output_path = Path(directory) / "output"
            environment = {
                "GITHUB_REPOSITORY": REPO,
                "GH_TOKEN": "test-token",
                "PR_NUMBER": "2989",
                "REVIEW_HEAD_SHA": HEAD,
                "GITHUB_RUN_ID": "100",
                "GITHUB_RUN_ATTEMPT": "1",
                "RESULT_PATH": str(result_path),
                "GITHUB_OUTPUT": str(output_path),
            }
            expected = FakeGitHub().result

            with patch.object(bridge, "collect", return_value=expected):
                result = CliRunner().invoke(
                    bridge.main,
                    args=["collect"],
                    env=environment,
                )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result_path.read_text()), expected)
            self.assertEqual(output_path.read_text(), "ready=true\n")

    def test_notifier_failure_does_not_expose_exception_contents(self):
        """Fail clearly without logging credentials embedded in an exception."""
        with patch.object(
            bridge,
            "_notify_from_environment",
            side_effect=RuntimeError("sensitive-token-or-signed-url"),
        ):
            result = CliRunner().invoke(bridge.main, args=["notify"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("RuntimeError", result.output)
        self.assertNotIn("sensitive-token-or-signed-url", result.output)


if __name__ == "__main__":
    unittest.main()
