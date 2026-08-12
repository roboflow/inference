import sys
from pathlib import Path

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from build_processor_jobs import build_jobs, load_corpus  # noqa: E402

MANIFEST = BENCHMARK_DIR / "workflows" / "manifest.json"


def test_workflow_corpus_has_one_profile_per_provisional_class():
    profiles = load_corpus(MANIFEST)

    assert set(profiles) == {
        "cpu-blur",
        "single-detection",
        "detection-tracking",
        "dual-detection",
        "instance-segmentation",
    }
    assert {profile["provisionalClass"] for profile in profiles.values()} == {
        "light",
        "medium",
        "heavy",
        "exclusive",
    }
    for profile in profiles.values():
        specification = profile["specification"]
        assert specification["version"]
        assert any(item.get("name") == "image" for item in specification["inputs"])
        assert profile["imageOutput"] in {
            output.get("name") for output in specification["outputs"]
        }
        step_names = [step["name"] for step in specification["steps"]]
        assert len(step_names) == len(set(step_names))


def test_job_builder_can_repeat_shared_model_and_mix_profiles():
    profiles = load_corpus(MANIFEST)

    jobs = build_jobs(
        profiles,
        ["single-detection", "detection-tracking"],
        "rtsp://localhost:8554/{stream}",
        "stream",
        repeat=2,
    )

    assert len(jobs) == 4
    assert len({job["id"] for job in jobs}) == 4
    assert len({job["sourceUrl"] for job in jobs}) == 4
    assert [job["benchmarkProfile"] for job in jobs] == [
        "single-detection",
        "single-detection",
        "detection-tracking",
        "detection-tracking",
    ]
