"""Execute the notifier's eligibility shell with network-free GitHub responses."""

import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/maintainer-review-slack.yml"
GITHUB_STUB = r"""
gh() {
  case "$*" in
    *collaborators/author/permission*)
      if [ "$MOCK_STATUS" = transport ]; then
        echo 'connection failed' >&2
        return 1
      fi
      printf 'HTTP/2.0 %s\nContent-Type: application/json\r\n\r\n' "$MOCK_STATUS"
      if [ "$MOCK_STATUS" != 200 ]; then
        printf '{"message":"Lookup failed"}\n'
        return 1
      fi
      printf '%s\n' "$MOCK_PERMISSION"
      ;;
    *repos/roboflow/inference/pulls/3035*)
      echo called >> "$MOCK_PR_CALLS"
      printf '%s\n' '{"state":"open","draft":false,"base":{"ref":"main","repo":{"full_name":"roboflow/inference"}},"head":{"repo":{"full_name":"roboflow/inference"}}}'
      ;;
    *) echo 'Unexpected GitHub request' >&2; return 97 ;;
  esac
}
"""


class EligibilityTests(unittest.TestCase):
    """Verify HTTP error handling in the actual pre-queue workflow step."""

    def _run_gate(self, *, status="200", permission="write"):
        """Run the workflow shell against an isolated, mocked issue comment."""
        workflow = yaml.safe_load(WORKFLOW.read_text())
        script = workflow["jobs"]["eligibility"]["steps"][0]["run"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / "event.json"
            event.write_text(
                json.dumps(
                    {
                        "issue": {"number": 3035},
                        "comment": {"user": {"login": "author"}},
                    }
                )
            )
            output = root / "output"
            output.touch()
            calls = root / "pr-calls"
            result = subprocess.run(
                ["bash", "-c", GITHUB_STUB + script],
                env={
                    "PATH": os.environ["PATH"],
                    "GITHUB_EVENT_NAME": "issue_comment",
                    "GITHUB_EVENT_PATH": str(event),
                    "GITHUB_OUTPUT": str(output),
                    "REPO": "roboflow/inference",
                    "MOCK_STATUS": status,
                    "MOCK_PERMISSION": permission,
                    "MOCK_PR_CALLS": str(calls),
                },
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            gate_result = (result, output.read_text(), calls.exists())

        return gate_result

    def test_permission_denials_skip_before_pr_lookup(self):
        """Skip 403/404 without producing queue output or fetching the PR."""
        for status in ("403", "404"):
            with self.subTest(status=status):
                result, output, fetched_pr = self._run_gate(status=status)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"HTTP {status}", result.stdout)
                self.assertEqual(output, "")
                self.assertFalse(fetched_pr)

    def test_unexpected_failures_remain_visible(self):
        """Fail on authentication, rate limit, server, and transport errors."""
        for status in ("401", "429", "500", "502", "transport"):
            with self.subTest(status=status):
                result, output, fetched_pr = self._run_gate(status=status)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("::error::", result.stdout)
                self.assertEqual(output, "")
                self.assertFalse(fetched_pr)

    def test_only_write_access_enters_queue(self):
        """Preserve the existing allowlist when the lookup succeeds."""
        for permission in ("write", "maintain", "admin", "read", "triage", "none"):
            with self.subTest(permission=permission):
                result, output, fetched_pr = self._run_gate(permission=permission)
                allowed = permission in {"write", "maintain", "admin"}
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output, "number=3035\n" if allowed else "")
                self.assertEqual(fetched_pr, allowed)
