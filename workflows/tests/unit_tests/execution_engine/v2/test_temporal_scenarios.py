"""Run the temporal operator examples end to end and check their key evidence.

The examples assert their own observations; these tests re-check the user-facing
facts from the saved evidence, so a weakened example cannot pass silently.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "development" / "workflows-2.0").is_dir()
)
DEMO = REPOSITORY_ROOT / "development/workflows-2.0/05-temporal-operators/run_demo.py"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("temporal-scenarios")
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(output_dir / "matplotlib"),
        PYTHONPATH=os.pathsep.join(
            str(REPOSITORY_ROOT / part)
            for part in (".", "workflows", "inference_models", "stream_vision")
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(DEMO),
            "--case",
            "all",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    document = json.loads((output_dir / "evidence.json").read_text())
    document["_output_dir"] = str(output_dir)

    return document


def test_main_graph_aligns_collects_selects_and_recollects(evidence):
    rig = evidence["rig"]

    assert rig["paired_image_pts"] == [[0], [80], [200]]
    assert rig["best_pts"] == [[80, 45], [240, 165]]
    assert rig["mosaic_last_pts"] == [[120, 125], [280, 285]]
    assert rig["recollected_pts"] == [120, 280, 125, 285]
    assert rig["layouts"]["frames"] == ["sample", "time"]
    assert rig["layouts"]["crops"] == ["sample", "static_nesting", "time"]
    assert rig["layouts"]["marker_crops"] == ["sample", "time", "dynamic_nesting"]
    assert rig["layouts"]["best"] == ["sample"]
    emitted = {name: counts["emitted"] for name, counts in rig["operators"].items()}
    assert emitted == {"rig": 8, "paired": 3, "clip": 2, "recollect": 1}
    assert rig["sources"]["sensor"]["processed"] == 3


def test_main_graph_writes_compiled_layout_results_and_gallery(evidence):
    case_dir = Path(evidence["_output_dir"]) / "rig"
    compiled = json.loads((case_dir / "compiled.json").read_text())
    results = json.loads((case_dir / "results.json").read_text())

    assert "clip" in json.dumps(compiled)
    assert {record["group"] for record in results} == {
        "paired",
        "rig_samples",
        "temperatures",
        "clips",
        "recollected",
    }
    assert (case_dir / "gallery" / "index.html").is_file()


def test_selected_frames_are_native_members_with_their_own_timestamps(evidence):
    results_path = Path(evidence["_output_dir"]) / "rig" / "results.json"
    results = json.loads(results_path.read_text())
    clips = [record for record in results if record["group"] == "clips"]
    best = clips[0]["fields"]["best"]
    frames = clips[0]["fields"]["frames"]

    assert [leaf["value"]["image_id"] for leaf in best["leaves"]] == [
        "left@80ms",
        "right@45ms",
    ]
    assert [leaf["source"] for leaf in best["leaves"]] == ["left", "right"]
    assert frames["root_pts_ms"] is None
    assert [leaf["index"] for leaf in frames["leaves"]][:2] == [[0, 0], [0, 1]]
    assert len(clips[0]["causes"]) == 4


def test_window_termination_policies(evidence):
    dropped = evidence["eof-drop"]
    emitted = evidence["eof-partial"]

    assert len(dropped["windows"]) == 2
    assert dropped["clip"]["partial_dropped"] == 1
    assert emitted["windows"][-1] == [240, 280, 245, 285]
    assert emitted["clip"]["partial_dropped"] == 0
    assert emitted["clip"]["emitted"] == 3


def test_explicit_context_policies_and_singleton_selection(evidence):
    rows = evidence["policies"]["rows"]
    selection = evidence["selection-k1"]

    assert rows[0] == {
        "first": [0, 5],
        "last": [80, 85],
        "best": [80, 45],
        "count": [None, None],
    }
    assert selection["brightest_pts"] == [[120, 5], [280, 165]]
    assert "a selected collection" in selection["recollect_rejected"]["message"]


def test_alignment_missing_and_late_inputs(evidence):
    assert evidence["missing-drop"]["rig"]["dropped"] == 1
    assert evidence["missing-drop"]["rig"]["emitted"] == 7
    assert evidence["missing-partial"]["gap"]["filtered"] == [[1]]
    assert len(evidence["missing-partial"]["pairs"]) == 8
    assert evidence["late"]["rig"]["late"] == 1
    assert evidence["late"]["pairs"][2] == [80, 85]


def test_alignment_errors_name_the_operator(evidence):
    clock = evidence["clock-error"]["error"]
    bound = evidence["capacity"]["bound_4_error"]

    assert clock["stage"] == "operator" and clock["operator"] == "rig"
    assert "other-media" in clock["message"]
    assert bound["stage"] == "operator" and bound["operator"] == "rig"
    assert len(evidence["capacity"]["default_bound_pairs"]) == 8


def test_admission_bound_one_still_collects_windows(evidence):
    admission = evidence["admission-one"]

    assert len(admission["windows"]) == 2
    for counts in admission["sources"].values():
        assert counts["admitted"] == counts["processed"] == 8


@pytest.mark.parametrize(
    ("case", "error_type", "reason"),
    [
        (
            "invalid-dynamic-crop",
            "OperatorInputError",
            "collect.crops ($steps.crop.crops): axis 'crop:regions' comes from "
            "$steps.crop.crops",
        ),
        (
            "invalid-second-t",
            "OperatorInputError",
            "collect.clips ($operators.clip.frames): axis 'operators.clip:t' comes "
            "from $operators.clip",
        ),
        (
            "invalid-middle-t",
            "LineageError",
            "is dynamic_nesting, after the time axis; a T-oriented collapse consumes "
            "only a final time axis. Collapse the axes after T first",
        ),
        (
            "invalid-lineage",
            "LineageError",
            "joins independent pulse domains (operator 'clip' via "
            "$operators.clip.frames; operator 'rig' via $operators.rig.samples); "
            "operator 'rig' feeds operator 'clip'",
        ),
        ("operator-in-child", "NestedWorkflowError", "declares operators ['clip']"),
    ],
)
def test_invalid_definitions_are_rejected_for_the_stated_reason(
    evidence, case, error_type, reason
):
    rejection = evidence[case]

    assert rejection["type"] == error_type
    assert reason in rejection["message"]


def test_gates_and_child_workflows_around_operators(evidence):
    nested = evidence["nested"]

    assert nested["left_statuses"].count("filtered") == 2
    assert all(finished["filtered"] == [[1]] for finished in nested["finished"])
    assert nested["operators"]["rig"]["emitted"] == 6


def test_stop_failure_and_restart(evidence):
    stop = evidence["stop"]
    failure = evidence["failure"]
    restart = evidence["restart"]

    assert 1 <= len(stop["pairs"]) < 8
    assert all(counts["closed"] for counts in stop["operators"].values())
    assert failure["error"]["stage"] == "step"
    assert failure["error"]["operator"] == "clip"
    assert failure["error"]["pulse"] == 1
    assert all(counts["closed"] for counts in failure["operators"].values())
    assert len(restart["windows"]) == 2


def test_passive_timestamped_window(evidence):
    passive = evidence["passive"]

    assert [leaf["pts_ms"] for leaf in passive["fields"]["best"]["leaves"]] == [80, 45]
    assert "need sources" in passive["operator_rejected"]["message"]


def test_the_lineage_rejection_suggests_a_held_parent_reference(evidence):
    message = evidence["invalid-lineage"]["message"]

    assert "for a window, a hold input: a held parent reference" in message
    assert "alignment" not in message
