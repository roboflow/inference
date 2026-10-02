"""Plugin example: catalogue module, session-scoped Factory, output conversion
and workload discovery through nested inputs.

The blocks come from ``example_plugin.py``, loaded by module name exactly as
an installed plugin would be.
"""

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.introspection import discover_workload
from v2_runner import plain


def plugin_catalogue(report, *, catalogue, load) -> None:
    """Load a plugin, share a session-scoped resource, convert outputs, list workload."""
    plugin = Catalogue.from_modules(["example_plugin"])
    report.check(
        "the plugin module exposes its blocks",
        sorted(plugin.block_types),
        ["plugin/ledger@v1", "plugin/load_model@v1", "plugin/locate@v1"],
    )

    plan = compile_workflow(load("plugin_resources"), catalogue=plugin)
    session = plan.create_session()
    rows = [session.run({"x": x}).rows()[0] for x in (1, 2)]
    report.check(
        "Factory(session): both ledger steps append to one list, across runs",
        [(row["size_a"], row["size_b"]) for row in rows],
        [(1, 2), (3, 4)],
    )
    ledgers = [session.instances[(name,)].ledger for name in ("ledger_a", "ledger_b")]
    report.check(
        "the two steps hold the same list, provided by the plugin catalogue",
        (ledgers[0] is ledgers[1], session.resources[("ledger_a",)]["ledger"].source),
        (True, "catalogue:example_plugin.ledger"),
    )
    fresh = plan.create_session()
    report.check(
        "a new session gets a new list",
        (
            fresh.run({"x": 1}).rows()[0]["size_b"],
            fresh.instances[("ledger_a",)].ledger is ledgers[0],
        ),
        (2, False),
    )
    report.check(
        "the point kind converts per output: own coordinates, or parent with the origin added",
        (rows[0]["own"], rows[0]["parent"]),
        ({"x": 1, "y": 2}, {"x": 101, "y": 202}),
    )

    workload_plan = compile_workflow(load("plugin_workload"), catalogue=plugin)
    workload = discover_workload(workload_plan)
    report.details["workload"] = plain(
        {
            "resources": [item.model_dump() for item in workload.resources.items],
            "complete": workload.resources.complete,
            "unknown": [
                problem.description for problem in workload.resources.unknown_reasons
            ],
        }
    )
    report.check(
        "workload: a child default and a bound literal are known models",
        [
            (item.identifier, item.used_by_steps)
            for item in sorted(
                workload.resources.items, key=lambda item: item.identifier
            )
        ],
        [
            ("large-model", ["$steps.literal_model/load"]),
            ("small-model", ["$steps.default_model/load"]),
        ],
    )
    report.check(
        "workload: the selected model is reported as unknown, naming its selector",
        (
            workload.resources.complete,
            [
                (problem.code.value, problem.details["selector"])
                for problem in workload.resources.unknown_reasons
            ],
        ),
        (False, [("unresolved_selector", "$inputs.model")]),
    )
    report.check(
        "running it loads each model as declared",
        workload_plan.create_session().run({"model": "custom-model"}).rows(),
        [
            {
                "default_model": "loaded small-model",
                "literal_model": "loaded large-model",
                "selected_model": "loaded custom-model",
            }
        ],
    )
