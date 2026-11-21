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
    """Exercise live-runner validation and reporting without warehouse credentials."""

    def setUp(self):
        """Load a fresh runner module and replace fixture-view creation for each test."""

        def _fixture(statement):
            marker = runner.re.search(r"/\* (rf_live_slow_[a-z0-9_]+) \*/", statement)
            name = marker.group(1) if marker else "fixture_view"
            columns = (
                [f"note_{i}" for i in range(17)]
                if "AS note_" in statement
                else ["id", "note"]
            )
            result = "catalog.schema." + name, columns

            return result

        patcher = patch.object(runner, "create_fixture_view", side_effect=_fixture)
        self.create_view = patcher.start()
        self.addCleanup(patcher.stop)

    def test_dispatch_values_are_passed_literally_without_shell_execution(self):
        """Verify dispatch values are passed literally without shell execution."""
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
        """Verify report redacts credentials before truncating."""
        with patch.dict(runner.os.environ, {"DATABRICKS_TOKEN": "private-token"}):
            self.assertEqual(
                runner._safe_error(RuntimeError("private-token failed")),
                "RuntimeError: *** failed",
            )

    def test_server_destination_accepts_only_https_origin(self):
        """Verify server destination accepts only https origin."""
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
        """Verify server uses operator destination and refuses redirects."""
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
                target.run({}, inputs={"token": "private-token"})
        self.assertEqual(
            post.call_args.args[0], "https://staging.example.com/workflows/run"
        )
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)
        self.assertEqual(post.call_count, 1)

    def test_server_error_does_not_truncate_credential_before_report_redaction(self):
        """Verify server error does not truncate credential before report redaction."""
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
                target.run({}, inputs={"token": token})
            self.assertIn(token, str(error.exception))
            detail = runner._safe_error(error.exception)
        self.assertNotIn("private-databricks", detail)

    def test_budget_error_redacts_credential_before_report_truncation(self):
        """Verify budget error redacts credential before report truncation."""
        token = "private-databricks-token-1234567890"
        target = Mock()
        target.run.side_effect = runner.RunError("x" * 270 + token, 0.1)
        with patch.object(runner, "env", return_value="configured"), patch.dict(
            runner.os.environ, {"DATABRICKS_TOKEN": token}
        ):
            with self.assertRaises(AssertionError) as error:
                runner.case_budget(target, table="catalog.schema.images", token=token)
            detail = runner._safe_error(error.exception)
        self.assertNotIn("private-databricks", detail)
        self.assertIn("***", detail)

    def test_history_refuses_redirects(self):
        """Verify history refuses redirects."""
        with patch.dict(
            runner.os.environ, {"DATABRICKS_SERVER_HOSTNAME": "warehouse.example.com"}
        ), patch.object(
            runner.requests, "get", return_value=Mock(status_code=307)
        ) as get:
            with self.assertRaises(RuntimeError):
                runner.query_history(["RUNNING"], since_ms=0, token="token")
        self.assertIs(get.call_args.kwargs["allow_redirects"], False)

    def test_unknown_or_empty_cases_fail_before_credentials_or_setup(self):
        """Verify unknown or empty cases fail before credentials or setup."""
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
        """Verify fixture table rejects sql fragments."""
        self.assertEqual(
            runner._fixture_table("catalog.schema.images"), "catalog.schema.images"
        )
        for table in ("images", "a.b.c;DROP TABLE other", "a.b.c --", "a.b.`c`"):
            with self.subTest(table=table), self.assertRaises(ValueError):
                runner._fixture_table(table)

    def test_metadata_checks_rejections_and_all_rejected_output(self):
        """Verify metadata checks row errors and the all-rejected contract."""
        target = Mock()
        target.run.side_effect = [
            {
                "names": [str(i) for i in range(runner.FIXTURE_ROWS)] + [None],
                "errors": [{"image_name": "901", "code": "METADATA_VALUE_TOO_LONG"}],
            },
            {
                "names": [str(i) for i in range(runner.FIXTURE_ROWS)] + [None],
                "errors": [{"code": "METADATA_TOO_LARGE"}],
            },
            {
                "names": [None],
                "metadata": [None],
                "shapes": [],
                "errors": [{"code": "METADATA_VALUE_TOO_LONG"}],
            },
        ]
        with patch.object(runner, "env", return_value="configured"):
            runner.case_big_metadata(
                target, table="catalog.schema.images", token="token"
            )
        self.assertEqual(target.run.call_count, 3)
        query = self.create_view.call_args_list[1].args[0]
        self.assertNotIn(
            "sql_statement", target.run.call_args_list[1].args[0]["steps"][0]
        )
        self.assertEqual(query.count("repeat('m', 4096)"), 17)

    def _slow_case(
        self,
        *,
        status="CANCELED",
        message="",
        observe=True,
        terminal_id="reader-select",
    ):
        self.create_view.reset_mock()
        queried = threading.Event()
        observed = threading.Event()
        probe = {"user_id": 42, "warehouse_id": "warehouse", "query_end_time_ms": 1000}

        def _run(specification, *, inputs):
            queried.set()
            self.assertTrue(observed.wait(timeout=3))
            raise runner.RunError("Statement has timed out after 7 seconds.", 0.01)

        def _probe(token):
            self.assertEqual(self.create_view.call_count, 1)
            self.assertFalse(queried.is_set())

            return probe

        def _history(statuses, *, since_ms, token, reader_probe):
            self.assertIs(reader_probe, probe)
            self.assertEqual(since_ms, probe["query_end_time_ms"])
            self.assertTrue(queried.wait(timeout=3))
            rows = [
                {"query_text": "CREATE VIEW /* rf_live_slow */", "status": "FINISHED"}
            ]
            if observe:
                active = not observed.is_set()
                rows.append(
                    {
                        "query_id": "reader-select" if active else terminal_id,
                        "query_text": "SELECT IDENTIFIER(:column_0), IDENTIFIER(:column_1) "
                        "FROM IDENTIFIER(:source) LIMIT :row_limit",
                        "status": "RUNNING" if active else status,
                        "error_message": message,
                    }
                )

            return rows

        original_wait = threading.Event.wait

        def _wait(event, timeout=None):
            if timeout == 0.5:
                observed.set()
            result = original_wait(event, timeout)

            return result

        with patch.object(runner, "env", return_value="configured"), patch.object(
            runner, "query_history", side_effect=_history
        ), patch.object(
            runner, "_reader_history_probe", side_effect=_probe
        ), patch.object(
            threading.Event, "wait", _wait
        ), patch.dict(
            runner._SLOW_STATEMENTS, clear=True
        ):
            result = runner.case_slow(
                Mock(run=_run),
                table="catalog.schema.images",
                token="token",
                deadline=5,
                extra_sql="",
            )
            outcome = runner._SLOW_STATEMENTS[result["query_id"]]

        return outcome

    def test_slow_case_distinguishes_cancellation_from_warehouse_timeout(self):
        """Verify the recorded outcomes for the same parameterized reader statement."""
        self.assertEqual(self._slow_case(), "canceled")
        self.assertEqual(
            self._slow_case(
                status="FAILED", message="Statement has timed out after 7 seconds."
            ),
            "warehouse_timeout",
        )

    def test_slow_case_rejects_unrelated_terminal_results(self):
        """Require warehouse evidence of cancellation or a specific statement timeout."""
        for status, message in (("FAILED", "permission denied"), ("FINISHED", "")):
            with self.subTest(status=status), self.assertRaisesRegex(
                AssertionError, "unexpected warehouse result"
            ):
                self._slow_case(status=status, message=message)

    def test_slow_case_requires_active_observation(self):
        """Verify slow case requires active observation."""
        with self.assertRaisesRegex(AssertionError, "not observed RUNNING"):
            self._slow_case(observe=False)

    def test_slow_case_rejects_a_different_terminal_statement(self):
        """Require the terminal state to belong to the observed reader statement."""
        with self.assertRaisesRegex(AssertionError, "statement ID changed"):
            self._slow_case(terminal_id="other-reader-select")

    def test_slow_history_rejects_ambiguous_statements(self):
        """Refuse to attribute two concurrent identical reads to the test."""
        rows = [
            {"query_text": runner._SLOW_READER_QUERY, "query_id": query_id}
            for query_id in ("reader-select", "another-select")
        ]
        with patch.object(runner, "query_history", return_value=rows):
            with self.assertRaisesRegex(AssertionError, "multiple reader statements"):
                runner._slow_reader_statement(
                    {"query_end_time_ms": 1000}, token="token"
                )

    def test_slow_history_filters_by_reader_and_warehouse_and_rejects_partial_page(
        self,
    ):
        """Verify filters sent to Databricks and require complete history."""
        response = Mock(status_code=200)
        response.json.return_value = {"has_next_page": True, "res": []}
        probe = {"user_id": 42, "warehouse_id": "warehouse"}
        with patch.object(runner, "env", return_value="configured"), patch.object(
            runner.requests, "get", return_value=response
        ) as get:
            with self.assertRaisesRegex(RuntimeError, "Incomplete reader history"):
                runner.query_history(
                    [], since_ms=1000, token="token", reader_probe=probe
                )

        params = get.call_args.kwargs["params"]
        self.assertEqual(params["filter_by.user_ids"], 42)
        self.assertEqual(params["filter_by.warehouse_ids"], "warehouse")
        self.assertEqual(params["filter_by.query_start_time_range.start_time_ms"], 1000)


if __name__ == "__main__":
    unittest.main()
