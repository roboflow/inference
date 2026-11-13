"""Authorization tests for manual hosted E2E dispatches."""

import os
import unittest
from unittest.mock import patch

import hosted_e2e_gate as gate


class GateTests(unittest.TestCase):
    """Accept any same-repository manual dispatch; reject everything else."""

    def setUp(self):
        """Use a synthetic manual dispatch of a feature branch."""
        self.env = {
            "GITHUB_REPOSITORY": "roboflow/inference",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/feat/anything",
            "GITHUB_SHA": "a" * 40,
            "HOSTED_E2E_TARGET": "staging",
        }

    def test_any_branch_or_tag_is_accepted_without_network(self):
        """Keep pre-sharding behavior: main, feature, fast-track and tag refs."""
        for ref in [
            "refs/heads/main",
            "refs/heads/feat/anything",
            "refs/heads/fast-track/post-v1.8.0-pt1",
            "refs/tags/v1.8.0",
        ]:
            with self.subTest(ref=ref), patch.dict(
                os.environ, {**self.env, "GITHUB_REF": ref}
            ):
                self.assertEqual(gate._authorize(), (ref, "a" * 40))

    def test_fork_non_manual_invalid_sha_target_or_ref_is_rejected(self):
        """Fail closed on anything that is not a manual same-repository commit."""
        for changed in [
            {"GITHUB_REPOSITORY": "someone/inference"},
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"GITHUB_SHA": "main; false"},
            {"HOSTED_E2E_TARGET": "other"},
            {"GITHUB_REF": "refs/pull/1/merge"},
            {"GITHUB_REF": "feat/anything"},
        ]:
            with self.subTest(changed=changed), patch.dict(
                os.environ, {**self.env, **changed}
            ), self.assertRaises(ValueError):
                gate._authorize()


if __name__ == "__main__":
    unittest.main()
