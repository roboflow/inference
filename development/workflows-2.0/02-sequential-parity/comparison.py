"""Compare a live V1 observation of a reference case with its V2 observation.

``match`` runs: V2 must reproduce the V1 rows, per-step call counts, per-step
argument/result sequences, error category, resolver calls, error-hook calls,
resources and instance reuse.

Other runs: V2 must meet the explicit expectation in ``parity.py``; the V1
observation is shown next to it and never counted as parity.

Only these normalizations are applied:

* step names: V2 path ``("child", "echo")`` is written ``child__echo`` as V1 does;
* numbers compare by value (V1 coerces ``float`` inputs, so ``1`` equals ``1.0``;
  a bool never equals a number);
* error classes compare through ``parity.ERROR_CATEGORIES``;
* calls of different steps are not ordered against each other (V1 does not
  define sibling order); calls of one step keep their order;
* an expanding V1 block returns one mapping per child; the V2 block returns one
  ``Batch`` per output. V1 results of steps whose V2 output declares ``expand``
  are rewritten to that form (same values, local indices ``(0,)``, ``(1,)``...);
* V2's observer reports a call when it returns: arguments after any in-place
  change and futures already resolved. V1 calls are read at the same moment
  (``arguments_after_call``, ``result_resolved``). A call that raised compares
  only the raised class, because V2 reports it through ``on_error``;
* V1 core blocks translated to fixtures compare their meaningful parts:
  ContinueIf value and whether it admitted; SwitchCase value and route;
* resource origin ``plugin_initializer`` (V1) equals ``catalogue_provider`` (V2).
"""

from typing import Any, Dict, List, Optional

from parity import ERROR_CATEGORIES, V2_EXPECTATIONS
from reference.catalogue import MATCH, ReferenceCase

SAME = "same"
DIFFERENT = "different"
BOUNDARY = "boundary_difference"


def equal(left: Any, right: Any) -> bool:
    """Compare JSON data; numbers by value, bools only with bools.

    Args:
        left: First value.
        right: Second value.

    Returns:
        Whether both values are semantically equal.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))

    return left == right


def error_category(error: Optional[Dict[str, Any]]) -> Optional[str]:
    """Map an error description to its category.

    Args:
        error: Error description from either runner, or ``None``.

    Returns:
        The category, ``unmapped:<class>`` for an unknown class, or ``None``.
    """
    if error is None:
        return None

    category = ERROR_CATEGORIES.get(error["type"], f"unmapped:{error['type']}")

    return category


def _v1_step_types(v1: Dict[str, Any]) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for session in v1["sessions"]:
        for name, compiled in (session.get("compiled_steps") or {}).items():
            types[name] = compiled["type"]

    return types


def _as_expanded_outputs(result: Any, outputs: List[str]) -> Any:
    if not outputs or not isinstance(result, list):
        return result

    expanded = {
        name: {
            "batch": [child[name] for child in result],
            "indices": [[position] for position in range(len(result))],
        }
        for name in outputs
    }

    return expanded


def _project_v1(
    entry: Dict[str, Any], step_type: str, expand_outputs: List[str]
) -> Dict[str, Any]:
    if "error" in entry:
        return {"raised": entry["error"]["type"]}

    arguments = entry.get("arguments_after_call", entry["arguments"])
    result = entry.get("result_resolved", entry.get("result"))
    result = _as_expanded_outputs(result, expand_outputs)
    if step_type == "roboflow_core/continue_if@v1":
        return {
            "value": arguments["evaluation_parameters"]["value"],
            "admitted": bool((result or {}).get("context")),
        }
    if step_type == "roboflow_core/switch_case@v1":
        targets = (result or {}).get("context") or []
        return {"value": arguments["value"], "route": sorted(targets)}

    return {"arguments": arguments, "result": result}


def _project_v2(entry: Dict[str, Any], v1_step_type: Optional[str]) -> Dict[str, Any]:
    if "raised" in entry:
        return {"raised": entry["raised"]}

    arguments, result = entry["arguments"], entry.get("result")
    if v1_step_type == "roboflow_core/continue_if@v1":
        return {"value": arguments["value"], "admitted": bool(result["select"])}
    if v1_step_type == "roboflow_core/switch_case@v1":
        return {"value": arguments["value"], "route": sorted(result["select"])}

    return {"arguments": arguments, "result": result}


def _calls_by_step(
    invocations: List[Dict[str, Any]], run: int, project
) -> Dict[str, List[Any]]:
    by_step: Dict[str, List[Any]] = {}
    for entry in invocations:
        if entry["run"] == run:
            by_step.setdefault(entry["step"], []).append(project(entry))

    return by_step


def _runs(observation: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    runs = {}
    for session in observation["sessions"]:
        for run in session["runs"]:
            runs[run["run"]] = run

    return runs


def _check(name: str, same: bool, **details: Any) -> Dict[str, Any]:
    check = {"check": name, "status": SAME if same else DIFFERENT, **details}

    return check


def _compare_match_run(
    number: int,
    v1_run: Dict[str, Any],
    v2_run: Dict[str, Any],
    *,
    v1: Dict[str, Any],
    v2: Dict[str, Any],
) -> List[Dict[str, Any]]:
    step_types = _v1_step_types(v1)
    expand_outputs = v2.get("expand_outputs", {})
    v1_calls = _calls_by_step(
        v1["invocations"],
        number,
        lambda e: _project_v1(
            e, step_types.get(e["step"], ""), expand_outputs.get(e["step"], [])
        ),
    )
    v2_calls = _calls_by_step(
        v2["invocations"], number, lambda e: _project_v2(e, step_types.get(e["step"]))
    )
    checks = [
        _check("rows", equal(v1_run["rows"], v2_run["rows"])),
        _check(
            "call_counts",
            equal(v1_run["call_counts"], v2_run["call_counts"]),
            v1=v1_run["call_counts"],
            v2=v2_run["call_counts"],
        ),
        _check(
            "error_category",
            error_category(v1_run["error"]) == error_category(v2_run["error"]),
            v1=error_category(v1_run["error"]),
            v2=error_category(v2_run["error"]),
        ),
    ]
    for step in sorted(set(v1_calls) | set(v2_calls)):
        checks.append(
            _check(
                f"calls:{step}",
                equal(v1_calls.get(step, []), v2_calls.get(step, [])),
            )
        )

    return checks


def _compare_inputs(v1_run: Dict[str, Any], v2_run: Dict[str, Any]) -> Dict[str, Any]:
    if equal(v1_run["inputs_after_run"], v2_run["inputs_after_run"]):
        return _check("inputs_after_run", True)
    if equal(v2_run["inputs_after_run"], v2_run["inputs"]):
        # V1 rewrote the caller's mapping during preparation; V2 did not.
        return {
            "check": "inputs_after_run",
            "status": BOUNDARY,
            "label": "D012-INPUT-PREPARATION",
            "v1": v1_run["inputs_after_run"],
            "v2": v2_run["inputs_after_run"],
        }

    return _check(
        "inputs_after_run",
        False,
        v1=v1_run["inputs_after_run"],
        v2=v2_run["inputs_after_run"],
    )


def _compare_expected_run(
    key: tuple, v2_run: Dict[str, Any], *, v2: Dict[str, Any]
) -> List[Dict[str, Any]]:
    expected = V2_EXPECTATIONS[key]
    number = key[1]
    checks = [
        _check("v2_rows", equal(expected.rows, v2_run["rows"]), expected=expected.rows),
        _check(
            "v2_call_counts",
            equal(expected.calls, v2_run["call_counts"]),
            expected=expected.calls,
            v2=v2_run["call_counts"],
        ),
        _check(
            "v2_error_category",
            expected.error_category == error_category(v2_run["error"]),
            expected=expected.error_category,
            v2=error_category(v2_run["error"]),
        ),
    ]
    for step, indices in (expected.indices or {}).items():
        actual = [
            entry["index"]
            for entry in v2["invocations"]
            if entry["run"] == number and entry["step"] == step
        ]
        checks.append(
            _check(f"v2_indices:{step}", actual == indices, expected=indices, v2=actual)
        )

    return checks


def _canonical_instances(
    invocations: List[Dict[str, Any]], match_runs: set
) -> Dict[str, List[int]]:
    # Rename instance identities by first appearance per (session, step), so
    # the two engines' ordinal schemes compare structurally.
    canonical: Dict[Any, int] = {}
    by_step: Dict[str, List[int]] = {}
    compared = [entry for entry in invocations if entry["run"] in match_runs]
    ordered = sorted(compared, key=lambda e: (e["session"], e["step"]))
    for entry in ordered:
        identity = canonical.setdefault(entry["instance"], len(canonical))
        runs = by_step.setdefault(entry["step"], [])
        if identity not in runs:
            runs.append(identity)

    return by_step


def _normalize_resources(resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    origins = {"plugin_initializer": "provider", "catalogue_provider": "provider"}
    normalized = [
        {
            "origin": origins.get(item["origin"], item["origin"]),
            "events": item["events"],
        }
        for item in resources
    ]

    return sorted(normalized, key=lambda item: repr(item))


def _case_level_checks(
    v1: Dict[str, Any], v2: Dict[str, Any], *, match_runs: set
) -> List[Dict[str, Any]]:
    v1_hooks = [
        {"step": call["step"], "cause_type": call["type"]}
        for call in v1["step_error_hook_calls"]
    ]
    v2_hooks = [
        {"step": call["step"], "cause_type": call["cause_type"]}
        for call in v2["error_hook_calls"]
    ]
    v1_bindings = [session.get("resource_bindings") or {} for session in v1["sessions"]]
    v2_bindings = [session.get("resource_bindings") or {} for session in v2["sessions"]]
    checks = [
        _check(
            "resolver_calls",
            equal(v1["resolver_calls"], v2["resolver_calls"]),
            v1=v1["resolver_calls"],
            v2=v2["resolver_calls"],
        ),
        _check("error_hook_calls", equal(v1_hooks, v2_hooks), v1=v1_hooks, v2=v2_hooks),
        _check(
            "resources",
            equal(
                _normalize_resources(v1["resources"]),
                _normalize_resources(v2["resources"]),
            ),
        ),
        _check(
            "resource_bindings",
            equal(v1_bindings, v2_bindings),
            v1=v1_bindings,
            v2=v2_bindings,
        ),
        _check(
            "instances_per_step",
            equal(
                _canonical_instances(v1["invocations"], match_runs),
                _canonical_instances(v2["invocations"], match_runs),
            ),
        ),
    ]

    return checks


def compare_case(
    case: ReferenceCase, v1: Dict[str, Any], v2: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare one case and decide its verdict.

    Args:
        case: Reference case.
        v1: Live V1 observation of the case.
        v2: V2 observation from ``v2_runner.observe_v2_case``.

    Returns:
        ``verdict`` (``parity``, ``difference_confirmed`` or
        ``unexpected_difference``), per-run checks and case-level checks.
    """
    v1_runs, v2_runs = _runs(v1), _runs(v2)
    runs = []
    all_match = True
    for number, run in enumerate(case.runs):
        comparison = run.comparison or case.comparison
        v1_run, v2_run = v1_runs[number], v2_runs[number]
        if comparison.kind == MATCH:
            checks = _compare_match_run(number, v1_run, v2_run, v1=v1, v2=v2)
        else:
            all_match = False
            checks = _compare_expected_run((case.case_id, number), v2_run, v2=v2)
            checks.append(
                {
                    "check": "v1_differs_from_v2",
                    "status": "info",
                    "value": not (
                        equal(v1_run["rows"], v2_run["rows"])
                        and equal(v1_run["call_counts"], v2_run["call_counts"])
                    ),
                }
            )
        checks.append(_compare_inputs(v1_run, v2_run))
        runs.append(
            {
                "run": number,
                "comparison": comparison.kind,
                "label": comparison.label,
                "checks": checks,
            }
        )

    match_runs = {
        number
        for number, run in enumerate(case.runs)
        if (run.comparison or case.comparison).kind == MATCH
    }
    case_checks = _case_level_checks(v1, v2, match_runs=match_runs)
    if not all_match:
        # Resources and hooks accumulate over every run, including runs that
        # deliberately differ; show them, but they cannot decide the verdict.
        for check in case_checks:
            if check["check"] != "resolver_calls" and check["status"] == DIFFERENT:
                check["status"] = "info"
    failed = [
        check["check"]
        for check in case_checks + [c for run in runs for c in run["checks"]]
        if check["status"] == DIFFERENT
    ]
    if failed:
        verdict = "unexpected_difference"
    elif all_match:
        verdict = "parity"
    else:
        verdict = "difference_confirmed"

    comparison_result = {
        "verdict": verdict,
        "failed_checks": failed,
        "runs": runs,
        "case_checks": case_checks,
    }

    return comparison_result
