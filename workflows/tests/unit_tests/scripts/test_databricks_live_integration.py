"""Offline checks of the credentialed runner; no warehouse or server is contacted."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/databricks_live_integration.py"
spec = importlib.util.spec_from_file_location("databricks_live_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LiveRunnerTest(unittest.TestCase):
    def setUp(self):
        def fixture(statement):
            marker = runner.re.search(r"/\* (rf_live_slow_[a-z0-9_]+) \*/", statement)
            name = marker.group(1) if marker else "fixture_view"
            columns = (
                [f"note_{i}" for i in range(17)]
                if "AS note_" in statement
                else ["id", "note"]
            )
            return "catalog.schema." + name, columns

        patcher = patch.object(runner, "create_fixture_view", side_effect=fixture)
        self.create_view = patcher.start()
        self.addCleanup(patcher.stop)

    def test_dispatch_values_are_passed_literally_without_shell_execution(self):
        workflow = (
            SCRIPT.parents[2] / ".github/workflows/databricks_live_integration.yml"
        )
        text = workflow.read_text()
        live_step = text.split("      - name: 🧪 Live warehouse cases", 1)[1]
        run_block = live_step.split("        run: |\n", 1)[1].split("      - name:", 1)[
            0
        ]
        shell = "\n".join(line[10:] for line in run_block.splitlines())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "workflows").mkdir()
            capture = directory / "arguments.json"
            fake_python = directory / "python"
            fake_python.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['CAPTURE'], 'w') as handle: json.dump(sys.argv[1:], handle)\n"
            )
            fake_python.chmod(0o700)
            hostile_cases = "import'; touch should-not-exist; echo '"
            environment = dict(
                os.environ,
                PATH=f"{directory}:{os.environ['PATH']}",
                CAPTURE=str(capture),
                LIVE_TARGET="server",
                LIVE_CASES=hostile_cases,
                LIVE_SETUP="true",
                LIVE_EXPECTATION="disabled",
            )
            subprocess.run(
                ["bash", "-eu", "-c", shell],
                cwd=directory,
                env=environment,
                check=True,
                capture_output=True,
            )
            arguments = json.loads(capture.read_text())
            self.assertEqual(arguments[arguments.index("--cases") + 1], hostile_cases)
            self.assertIn("--setup", arguments)
            self.assertIn("--expect-disabled", arguments)
            self.assertFalse((directory / "workflows/should-not-exist").exists())

    def test_report_redacts_credentials_before_truncating(self):
        with patch.dict(runner.os.environ, {"DATABRICKS_TOKEN": "private-token"}):
            self.assertEqual(
                runner._safe_error(RuntimeError("private-token failed")),
                "RuntimeError: *** failed",
            )

    def test_server_destination_accepts_only_https_origin(self):
        self.assertEqual(
            runner._server_origin("https://staging.example.com/"),
            "https://staging.example.com",
        )
        for value in (
            "",
            "http://staging.example.com",
            "//staging.example.com",
            "https://user:password@staging.example.com",
            "https://staging.example.com/path",
            "https://staging.example.com?secret=x",
            "https://staging.example.com#fragment",
            "https://staging.example.com:9001",
            "https://staging.example.com\n",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner._server_origin(value)

    def test_server_uses_operator_destination_and_refuses_redirects(self):
        response = Mock(status_code=307, text="redirect")
        with patch.dict(
            runner.os.environ,
            {
                "DATABRICKS_LIVE_BASE_URL": "https://staging.example.com",
                "ROBOFLOW_API_KEY": "private-key",
            },
        ), patch.object(runner.requests, "post", return_value=response) as post:
            target = runner.ServerTarget()
            with self.assertRaises(runner.RunError):
                target.run({}, {"token": "private-token"})
        self.assertEqual(
            post.call_args.args[0], "https://staging.example.com/workflows/run"
        )
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)
        self.assertEqual(post.call_count, 1)

    def test_server_error_does_not_truncate_credential_before_report_redaction(self):
        token = "private-databricks-token-1234567890"
        response = Mock(status_code=400, text="x" * 480 + token)
        with patch.dict(
            runner.os.environ,
            {
                "DATABRICKS_LIVE_BASE_URL": "https://staging.example.com",
                "ROBOFLOW_API_KEY": "private-key",
                "DATABRICKS_TOKEN": token,
            },
        ), patch.object(runner.requests, "post", return_value=response):
            target = runner.ServerTarget()
            with self.assertRaises(runner.RunError) as error:
                target.run({}, {"token": token})
            self.assertIn(token, str(error.exception))
            detail = runner._safe_error(error.exception)
        self.assertNotIn("private-databricks", detail)

    def test_budget_error_redacts_credential_before_report_truncation(self):
        token = "private-databricks-token-1234567890"
        target = Mock()
        target.run.side_effect = runner.RunError("x" * 270 + token, 0.1)
        with patch.object(runner, "env", return_value="configured"), patch.dict(
            runner.os.environ, {"DATABRICKS_TOKEN": token}
        ):
            with self.assertRaises(AssertionError) as error:
                runner.case_budget(target, "catalog.schema.images", token)
            detail = runner._safe_error(error.exception)
        self.assertNotIn("private-databricks", detail)
        self.assertIn("***", detail)

    def test_history_refuses_redirects(self):
        with patch.dict(
            runner.os.environ, {"DATABRICKS_SERVER_HOSTNAME": "warehouse.example.com"}
        ), patch.object(
            runner.requests, "get", return_value=Mock(status_code=307)
        ) as get:
            with self.assertRaises(RuntimeError):
                runner.query_history(["RUNNING"], 0, "token")
        self.assertIs(get.call_args.kwargs["allow_redirects"], False)

    def test_unknown_or_empty_cases_fail_before_credentials_or_setup(self):
        for cases in ("", "unknown", "import,unknown", "import,", "import;echo secret"):
            with self.subTest(cases=cases), patch.object(
                runner.sys, "argv", ["runner", "--cases", cases, "--setup"]
            ), patch.object(runner, "env") as env, patch.object(
                runner, "setup_fixtures"
            ) as setup:
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                env.assert_not_called()
                setup.assert_not_called()

    def test_fixture_table_rejects_sql_fragments(self):
        self.assertEqual(
            runner._fixture_table("catalog.schema.images"), "catalog.schema.images"
        )
        for table in ("images", "a.b.c;DROP TABLE other", "a.b.c --", "a.b.`c`"):
            with self.subTest(table=table), self.assertRaises(ValueError):
                runner._fixture_table(table)

    def test_metadata_checks_truncation_and_aggregate_overflow_separately(self):
        target = Mock()
        target.run.side_effect = [
            {"names": ["901"], "metadata": [{"note": "m" * 4096}]},
            {"names": [str(i) for i in range(runner.FIXTURE_ROWS)]},
        ]
        with patch.object(runner, "env", return_value="configured"):
            runner.case_big_metadata(target, "catalog.schema.images", "token")
        self.assertEqual(target.run.call_count, 2)
        query = self.create_view.call_args_list[1].args[0]
        self.assertNotIn(
            "sql_statement", target.run.call_args_list[1].args[0]["steps"][0]
        )
        self.assertEqual(query.count("repeat('m', 4096)"), 17)

    def _slow_case(self, *, status="CANCELED", observe=True):
        queried = threading.Event()
        observed = threading.Event()
        state = {}

        class Target:
            def run(self, specification, inputs):
                state["query"] = specification["steps"][0]["table"]
                queried.set()
                if not observed.wait(timeout=3):
                    raise AssertionError("test observer did not run")
                raise runner.RunError("query timeout", 0.01)

        def history(statuses, since_ms, token):
            self.assertTrue(queried.wait(timeout=3))
            if statuses == ["RUNNING"]:
                # Return RUNNING while the target is still executing. The
                # mocked Event.wait below releases the query after observation.
                return (
                    [{"query_text": state["query"], "status": "RUNNING"}]
                    if observe
                    else []
                )
            return []

        original_wait = threading.Event.wait

        def wait(event, timeout=None):
            if timeout == 0.5:
                observed.set()
            return original_wait(event, timeout)

        with patch.object(runner, "env", return_value="configured"), patch.object(
            runner, "query_history", side_effect=history
        ), patch.object(
            runner, "wait_for_history", return_value={"status": status}
        ), patch.object(
            threading.Event, "wait", wait
        ):
            result = runner.case_slow(Target(), "catalog.schema.images", "token", 5, "")
        return result

    def test_slow_case_observes_active_query_and_accepts_canceled(self):
        self.assertEqual(self._slow_case()["status"], "CANCELED")

    def test_slow_case_rejects_unrelated_failed_state(self):
        with self.assertRaisesRegex(AssertionError, "did not confirm cancellation"):
            self._slow_case(status="FAILED")

    def test_slow_case_requires_active_observation(self):
        with self.assertRaisesRegex(AssertionError, "not observed RUNNING"):
            self._slow_case(observe=False)


if __name__ == "__main__":
    unittest.main()
