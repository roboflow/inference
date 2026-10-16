"""Authorization tests for the existing release fast-track process."""

import copy
import os
import unittest
from unittest.mock import patch

import hosted_e2e_gate as gate


class GateTests(unittest.TestCase):
    """Require the exact approved PR revision and release ancestry."""

    def setUp(self):
        """Use a synthetic approved v1.8.0 fast-track umbrella PR."""
        self.env = {
            "GITHUB_REPOSITORY": "roboflow/inference",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/fast-track/post-v1.8.0-pt1",
            "GITHUB_SHA": "a" * 40,
            "HOSTED_E2E_DEFAULT_REF": "refs/heads/main",
            "HOSTED_E2E_TARGET": "staging",
        }
        self.pr = {
            "state": "open",
            "draft": False,
            "head": {
                "repo": {"full_name": "roboflow/inference"},
                "ref": "fast-track/post-v1.8.0-pt1",
                "sha": "a" * 40,
            },
            "base": {"ref": "main"},
            "labels": [{"name": "fast-track-deploy-approved"}],
        }
        self.compare = {"status": "ahead", "merge_base_commit": {"sha": "b" * 40}}
        self.tag = {"type": "commit", "sha": "b" * 40}

    def _get(self, path):
        if path.startswith("pulls?"):
            return [self.pr]
        if path.startswith("git/ref/tags/"):
            self.assertEqual(path, "git/ref/tags/v1.8.0")
            return {"object": self.tag}
        if path.startswith("git/tags/"):
            return {"object": {"type": "commit", "sha": "b" * 40}}
        self.assertEqual(path, f"compare/{'b' * 40}...{'a' * 40}")
        return self.compare

    def test_main_does_not_need_a_release_pr(self):
        """Keep existing default-branch manual execution available."""
        with patch.dict(
            os.environ, {**self.env, "GITHUB_REF": "refs/heads/main"}
        ), patch.object(gate, "_get") as get:
            self.assertEqual(gate._authorize(), ("refs/heads/main", "a" * 40))
            get.assert_not_called()

    def test_approved_fast_track_matches_sha_and_release_tag(self):
        """Allow approved release heads for lightweight and annotated tags."""
        with patch.dict(os.environ, self.env), patch.object(
            gate, "_get", side_effect=self._get
        ):
            self.assertEqual(gate._authorize(), (self.env["GITHUB_REF"], "a" * 40))
            self.tag = {"type": "tag", "sha": "c" * 40}
            self.assertEqual(gate._authorize(), (self.env["GITHUB_REF"], "a" * 40))

    def test_unapproved_draft_closed_foreign_or_moved_head_is_rejected(self):
        """A branch name alone must never authorize hosted credentials."""
        original = copy.deepcopy(self.pr)
        variants = [
            {"labels": []},
            {"draft": True},
            {"state": "closed"},
            {"head": {**original["head"], "sha": "c" * 40}},
            {"head": {**original["head"], "repo": {"full_name": "someone/inference"}}},
            {"base": {"ref": "other"}},
        ]
        for changed in variants:
            self.pr = {**original, **changed}
            with self.subTest(changed=changed), patch.dict(
                os.environ, self.env
            ), patch.object(gate, "_get", side_effect=self._get), self.assertRaises(
                ValueError
            ):
                gate._authorize()

    def test_wrong_release_ancestry_or_api_failure_is_rejected(self):
        """Fail closed if approval or release ancestry cannot be established."""
        for comparison in [
            {"status": "diverged", "merge_base_commit": {"sha": "b" * 40}},
            {"status": "ahead", "merge_base_commit": {"sha": "c" * 40}},
        ]:
            self.compare = comparison
            with patch.dict(os.environ, self.env), patch.object(
                gate, "_get", side_effect=self._get
            ), self.assertRaises(ValueError):
                gate._authorize()
        with patch.dict(os.environ, self.env), patch.object(
            gate, "_get", side_effect=OSError("unavailable")
        ), self.assertRaises(OSError):
            gate._authorize()

    def test_fork_pr_and_arbitrary_branch_are_rejected(self):
        """Reject non-manual events and branches outside the release convention."""
        for changed in [
            {"GITHUB_REPOSITORY": "someone/inference"},
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"GITHUB_REF": "refs/heads/feature"},
            {"GITHUB_REF": "refs/heads/fast-track/not-a-release"},
        ]:
            with patch.dict(os.environ, {**self.env, **changed}), patch.object(
                gate, "_get"
            ) as get, self.assertRaises(ValueError):
                gate._authorize()
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
