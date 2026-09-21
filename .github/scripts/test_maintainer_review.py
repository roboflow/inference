"""Network-free behavior tests for the privileged review handoff."""

import copy
import io
import json
import unittest
import urllib.error
import zipfile
from unittest.mock import Mock, patch

import maintainer_review as bridge

REPO = "roboflow/inference"
HEAD = "a" * 40
TIME = "2026-09-21T10:00:00Z"
LATER = "2026-09-21T10:01:00Z"


def comment(number, body, bot=True):
    return {
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


class FakeGitHub:
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
                f"<!-- claude-pr-review:run-marker -->\n<!-- claude-pr-review:run=100:1 -->\n🤖 **Claude review started** at commit `{HEAD}`.",
            ),
            comment(11, bridge.PASS),
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

    def api(self, path, data=None, method=None):
        if path == "pulls/2989":
            return copy.deepcopy(self.pr)
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
            value = comment(max(c["id"] for c in self.comments) + 1, data["body"])
            self.comments.append(value)
            return value
        raise AssertionError(f"Unexpected API request: {path}")

    def pages(self, path, key=None):
        if path == "issues/2989/comments":
            return self.comments
        if path == "actions/runs/100/artifacts":
            return self.artifacts
        raise AssertionError(path)

    def artifact(self, artifact_id):
        assert artifact_id == 200
        return self.result


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.github = FakeGitHub()
        self.event = {"workflow_run": {"id": 100, "run_attempt": 1}}

    def automatic(self):
        return bridge.automatic_handoff(self.github, self.event)

    def escalation(
        self,
        body="/maintainer-review I disagree with the backwards-compatibility finding.",
    ):
        request = comment(20, body, bot=False)
        self.github.comments.append(request)
        return {
            "action": "created",
            "issue": {"number": 2989, "pull_request": {}},
            "comment": copy.deepcopy(request),
        }

    def test_collect_and_notify_current_pass(self):
        self.assertEqual(
            bridge.collect(self.github, 2989, HEAD, 100, 1), self.github.result
        )
        self.assertEqual(self.automatic()["key"], f"pass:{HEAD}")

    def test_no_pass_for_findings_skip_or_missing_prompt(self):
        for body in (
            "Blocking correctness issue",
            "[review skipped — non-substantive change]",
            "",
        ):
            with self.subTest(body=body):
                self.github.comments[1]["body"] = body
                self.assertIsNone(bridge.collect(self.github, 2989, HEAD, 100, 1))
                self.assertIsNone(self.automatic())

    def test_success_without_result_is_not_approval(self):
        self.github.artifacts = []
        self.assertIsNone(self.automatic())

    def test_human_cannot_forge_pass(self):
        self.github.comments[1] = comment(11, bridge.PASS, bot=False)
        self.assertIsNone(self.automatic())

    def test_wrong_app_cannot_forge_pass(self):
        self.github.comments[1]["performed_via_github_app"] = {"id": 123}
        self.assertIsNone(self.automatic())

    def test_old_pass_cannot_be_reused(self):
        self.github.comments[1]["id"] = 9
        self.assertIsNone(self.automatic())

    def test_edited_pass_is_rejected(self):
        self.github.comments[1]["updated_at"] = LATER
        self.assertIsNone(self.automatic())

    def test_push_invalidates_approval(self):
        self.github.pr["head"]["sha"] = "b" * 40
        self.assertIsNone(self.automatic())

    def test_new_review_supersedes_older_pass(self):
        self.github.comments.append(
            comment(12, "<!-- claude-pr-review:run-marker -->\nNew review")
        )
        self.assertIsNone(self.automatic())

    def test_ineligible_pr_is_not_announced(self):
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
        self.github.run["event"] = "workflow_dispatch"
        self.github.run["pull_requests"] = []
        self.assertIsNotNone(self.automatic())

    def test_pass_outside_run_window_is_rejected(self):
        self.github.run["updated_at"] = "2026-09-20T00:00:00Z"
        self.assertIsNone(self.automatic())

    def test_wrong_attempt_repository_sha_or_type_is_rejected(self):
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
        result = bridge.escalation_handoff(self.github, self.escalation())
        self.assertEqual(result["key"], "escalation:20")
        self.assertTrue(result["reason"].startswith("I disagree"))

    def test_author_can_escalate_fork_and_opt_out(self):
        self.github.pr["head"]["repo"]["full_name"] = "author/fork"
        self.github.pr["labels"] = [{"name": "skip-claude-review"}]
        self.assertIsNotNone(bridge.escalation_handoff(self.github, self.escalation()))

    def test_empty_reason_or_embedded_command_is_ignored(self):
        for body in (
            "/maintainer-review",
            "/maintainer-review   ",
            "Please /maintainer-review why",
            "> /maintainer-review why",
        ):
            with self.subTest(body=body):
                self.github = FakeGitHub()
                self.assertIsNone(
                    bridge.escalation_handoff(self.github, self.escalation(body))
                )

    def test_unrelated_user_cannot_escalate_but_maintainer_can(self):
        event = self.escalation()
        self.github.pr["user"]["id"] = 999
        self.assertIsNone(bridge.escalation_handoff(self.github, event))
        self.github.permission = "write"
        self.assertIsNotNone(bridge.escalation_handoff(self.github, event))

    def test_deleted_or_edited_escalation_is_not_replayed(self):
        event = self.escalation()
        self.github.comments[-1]["body"] = "Never mind"
        self.assertIsNone(bridge.escalation_handoff(self.github, event))

    def test_bot_cannot_request_human_escalation(self):
        event = self.escalation()
        event["comment"]["user"]["type"] = "Bot"
        self.assertIsNone(bridge.escalation_handoff(self.github, event))

    def test_post_retry_is_noop_and_escalation_reuses_thread(self):
        post = Mock(return_value={"channel": "C123", "ts": "12345.000001"})
        handoff = self.automatic()
        self.assertTrue(
            bridge.publish(self.github, handoff, "C123", "U123,U456", "secret", post)
        )
        self.assertFalse(
            bridge.publish(self.github, handoff, "C123", "U123,U456", "secret", post)
        )
        self.assertEqual(post.call_count, 1)
        escalation = bridge.escalation_handoff(self.github, self.escalation())
        bridge.publish(self.github, escalation, "C123", "U123", "secret", post)
        self.assertEqual(post.call_args.args[1]["thread_ts"], "12345.000001")
        self.assertEqual(
            sum(bridge.STATE_MARKER in c["body"] for c in self.github.comments), 1
        )
        self.assertNotIn("secret", self.github.comments[2]["body"])
        self.assertIn(
            "https://roboflow.slack.com/archives/C123/p12345000001",
            self.github.comments[2]["body"],
        )

    def test_forged_state_cannot_redirect_message(self):
        state = {
            "repository": REPO,
            "pr_number": 2989,
            "channel": "C123",
            "thread_ts": "999.000001",
            "delivered": [],
        }
        forged = {
            "state": state,
            "signature": bridge.state_signature(state, "wrong-key"),
        }
        self.github.comments.append(
            comment(15, bridge.STATE_MARKER + json.dumps(forged) + " -->")
        )
        post = Mock(return_value={"channel": "C123", "ts": "123.000001"})
        bridge.publish(self.github, self.automatic(), "C123", "U123", "secret", post)
        self.assertNotIn("thread_ts", post.call_args.args[1])

    def test_untrusted_title_and_reason_are_plain_text(self):
        self.github.pr["title"] = "<!channel> $(curl attacker)"
        handoff = bridge.escalation_handoff(
            self.github, self.escalation("/maintainer-review <@U999> <!channel>")
        )
        post = Mock(return_value={"channel": "C123", "ts": "123.000001"})
        bridge.publish(self.github, handoff, "C123", "U123", "secret", post)
        blocks = post.call_args.args[1]["blocks"]
        self.assertEqual(blocks[1]["text"]["type"], "plain_text")
        self.assertEqual(blocks[-1]["text"]["type"], "plain_text")

    def test_push_while_waiting_does_not_post(self):
        handoff = self.automatic()
        self.github.pr["head"]["sha"] = "b" * 40
        post = Mock()
        self.assertFalse(
            bridge.publish(self.github, handoff, "C123", "U123", "secret", post)
        )
        post.assert_not_called()

    def test_slack_failure_does_not_record_delivery(self):
        post = Mock(side_effect=ValueError("Slack failed"))
        with self.assertRaises(ValueError):
            bridge.publish(
                self.github, self.automatic(), "C123", "U123", "secret", post
            )
        self.assertFalse(
            any(bridge.STATE_MARKER in c["body"] for c in self.github.comments)
        )

    def test_save_failure_surfaces_instead_of_claiming_success(self):
        self.github.fail_save = True
        post = Mock(return_value={"channel": "C123", "ts": "123.000001"})
        with self.assertRaises(RuntimeError):
            bridge.publish(
                self.github, self.automatic(), "C123", "U123", "secret", post
            )

    def test_bad_configuration_fails_before_posting(self):
        for channel, maintainers, token in (
            ("#release", "U123", "secret"),
            ("C123", "@everyone", "secret"),
            ("C123", "U123", ""),
        ):
            post = Mock()
            with self.assertRaises(ValueError):
                bridge.publish(
                    self.github, self.automatic(), channel, maintainers, token, post
                )
            post.assert_not_called()


class ArtifactTests(unittest.TestCase):
    def archive(self, name, content):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name, content)
        return raw.getvalue()

    def test_json_is_read_without_extracting(self):
        self.assertEqual(
            bridge.read_result(self.archive("result.json", '{"version":1}')),
            {"version": 1},
        )

    def test_path_traversal_and_large_content_are_rejected(self):
        for name, content in (("../result.json", "{}"), ("result.json", " " * 9000)):
            with self.assertRaises(ValueError):
                bridge.read_result(self.archive(name, content))

    def test_artifact_redirect_does_not_forward_github_token(self):
        github = bridge.GitHub(REPO, "github-secret")
        github.request = Mock(
            side_effect=urllib.error.HTTPError(
                "unused",
                302,
                "redirect",
                {"Location": "https://storage.example/result"},
                None,
            )
        )
        response = io.BytesIO(self.archive("result.json", "{}"))
        with patch("urllib.request.urlopen", return_value=response) as download:
            self.assertEqual(github.artifact(1), {})
            download.assert_called_once_with(
                "https://storage.example/result", timeout=30
            )


if __name__ == "__main__":
    unittest.main()
