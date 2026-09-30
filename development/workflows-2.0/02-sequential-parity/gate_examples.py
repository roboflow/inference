"""A gate on a nested workflow also governs the inputs the child forwards.

``workflows/forwarding_gates.json``: ``child`` has a body step and forwards
its batch input ``x``, its default ``tag`` and the bound literal ``note``.
``gate`` targets the whole child. ``consumer`` (declared first) reads the
forwarded value, ``recover`` accepts unavailable values and joins it with the
ungated ``sibling``.

``workflows/forwarding_gates_groups.json``: the child forwards groups; a
parent-level gate admits or denies whole groups, including a genuinely empty
one. Both reducers process admitted empty groups; ``collect`` also accepts
unavailable inputs, while ``total`` requires an available group.

V1 lets forwarded values bypass a gate on the child (measured by the nested
oracle); the V2 behaviour below is an intentional difference. These are not
among the 45 reference cases.

Edit ``X``, ``GROUPS`` or the masks below and rerun
``run_demo.py --scenario capabilities --case nested_forwarding_gates``. A mask
without an expected entry is recorded, not checked.
"""

from typing import Any, Dict, List

from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from v2_runner import InvocationRecorder, plain

X = [10, 20, 30]
SCALAR_MASKS = {
    "all_false": [False, False, False],
    "partial": [False, True, False],
    "all_true": [True, True, True],
}
ALL_INDICES = [[0], [1], [2]]
SCALAR_STEPS = ["child__body", "consumer", "recover", "sibling", "gate"]
SCALAR_EXPECTED = {
    "all_false": {
        "rows": [
            {
                "forwarded": None,
                "tag": None,
                "note": None,
                "consumed": None,
                "recovered": [None, x],
                "sibling": x,
                "original": x,
            }
            for x in X
        ],
        "calls": {
            "child__body": [],
            "consumer": [],
            "recover": [[0], [1], [2]],
            "sibling": [[0], [1], [2]],
            "gate": [[0], [1], [2]],
        },
    },
    "partial": {
        "rows": [
            {
                "forwarded": None,
                "tag": None,
                "note": None,
                "consumed": None,
                "recovered": [None, 10],
                "sibling": 10,
                "original": 10,
            },
            {
                "forwarded": 20,
                "tag": "tag-default",
                "note": "bound-literal",
                "consumed": 20,
                "recovered": [20, 20],
                "sibling": 20,
                "original": 20,
            },
            {
                "forwarded": None,
                "tag": None,
                "note": None,
                "consumed": None,
                "recovered": [None, 30],
                "sibling": 30,
                "original": 30,
            },
        ],
        "calls": {
            "child__body": [[1]],
            "consumer": [[1]],
            "recover": [[0], [1], [2]],
            "sibling": [[0], [1], [2]],
            "gate": [[0], [1], [2]],
        },
    },
    "all_true": {
        "rows": [
            {
                "forwarded": x,
                "tag": "tag-default",
                "note": "bound-literal",
                "consumed": x,
                "recovered": [x, x],
                "sibling": x,
                "original": x,
            }
            for x in X
        ],
        "calls": {step: ALL_INDICES for step in SCALAR_STEPS},
    },
}

GROUPS = [[1, 2], [], [3]]
GROUP_MASKS = {
    "deny_empty_group": [True, False, True],
    "admit_empty_group": [False, True, True],
}
GROUP_STEPS = ["child__body", "collect", "total"]
GROUP_EXPECTED = {
    "deny_empty_group": {
        "rows": [
            {"forwarded": [1, 2], "collected": [1, 2], "total": 3, "original": [1, 2]},
            {"forwarded": [], "collected": None, "total": None, "original": []},
            {"forwarded": [3], "collected": [3], "total": 3, "original": [3]},
        ],
        "filtered": [[1]],
        "calls": {
            "child__body": [[0, 0], [0, 1], [2, 0]],
            "collect": [[0], [2]],
            "total": [[0], [2]],
        },
    },
    "admit_empty_group": {
        "rows": [
            {"forwarded": [], "collected": None, "total": None, "original": [1, 2]},
            {"forwarded": [], "collected": [], "total": 0, "original": []},
            {"forwarded": [3], "collected": [3], "total": 3, "original": [3]},
        ],
        "filtered": [[0]],
        "calls": {"child__body": [[2, 0]], "collect": [[1], [2]], "total": [[1], [2]]},
    },
}


def _calls_by_step(recorder: InvocationRecorder, steps: List[str]) -> Dict[str, Any]:
    return {
        step: [
            entry["index"] for entry in recorder.invocations if entry["step"] == step
        ]
        for step in steps
    }


def _forwarded_filtered(result: Any) -> List[List[int]]:
    (entry,) = result.selections["forwarded"].values()

    return [list(index) for index in result.filtered_paths.get(entry, ())]


def _scalar_runs(report, plan) -> None:
    order = ["__".join(step.path) for step in plan.steps]
    report.details["scalar_step_order"] = order
    report.check(
        "the compiler runs the gate before the consumer declared ahead of it",
        order.index("gate") < order.index("consumer"),
        True,
    )
    report.details["scalar_runs"] = {}
    for name, mask in SCALAR_MASKS.items():
        recorder = InvocationRecorder()
        result = plan.create_session(observer=recorder).run({"x": X, "keep": mask})
        rows = plain(result.rows())
        calls = _calls_by_step(recorder, SCALAR_STEPS)
        report.details["scalar_runs"][name] = {
            "keep": mask,
            "rows": rows,
            "calls": calls,
        }
        expected = SCALAR_EXPECTED.get(name)
        if expected is None:
            continue
        report.check(
            f"[whole-child gate, V1 leaks] {name}: rows", rows, expected["rows"]
        )
        report.check(f"{name}: call indices per step", calls, expected["calls"])


def _group_runs(report, plan) -> None:
    report.details["group_runs"] = {}
    for name, mask in GROUP_MASKS.items():
        recorder = InvocationRecorder()
        result = plan.create_session(observer=recorder).run(
            {"groups": GROUPS, "keep": mask}
        )
        observed = {
            "rows": plain(result.rows()),
            "filtered": _forwarded_filtered(result),
            "calls": _calls_by_step(recorder, GROUP_STEPS),
        }
        report.details["group_runs"][name] = {"keep": mask, **observed}
        expected = GROUP_EXPECTED.get(name)
        if expected is None:
            continue
        report.check(
            f"[whole-child gate] {name}: rows, denied groups and call indices",
            observed,
            {key: expected[key] for key in ("rows", "filtered", "calls")},
        )


def nested_forwarding_gates(report, *, catalogue, load) -> None:
    """Gate a child that forwards inputs, a default, a literal and groups."""
    _scalar_runs(
        report, compile_workflow(load("forwarding_gates"), catalogue=catalogue)
    )
    _group_runs(
        report, compile_workflow(load("forwarding_gates_groups"), catalogue=catalogue)
    )
