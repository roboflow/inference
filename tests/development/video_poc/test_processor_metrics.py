import sys
from pathlib import Path

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from processor_metrics import ProcessorMetrics  # noqa: E402
from worker_lifecycle import schedule_retirement  # noqa: E402


def test_processor_image_includes_metrics_module():
    dockerfile = (PROCESSOR_DIR / "Dockerfile").read_text()

    assert "COPY processor_metrics.py /app/processor_metrics.py" in dockerfile
    assert "COPY worker_lifecycle.py /app/worker_lifecycle.py" in dockerfile


def test_processor_retires_after_metrics_grace_period():
    processor = (PROCESSOR_DIR / "processor.py").read_text()

    assert 'os.getenv("PROCESSOR_FINAL_METRICS_GRACE_S", "35")' in processor
    assert "schedule_retirement(" in processor
    assert "self._delete_retiring_pod" in processor


def test_retirement_waits_for_final_metrics_scrape_window():
    events = []

    class FakeTimer:
        def __init__(self, delay, callback):
            events.append(("created", delay))
            self.callback = callback
            self.daemon = False

        def start(self):
            events.append(("started", self.daemon))

    timer = schedule_retirement(
        35,
        lambda: events.append(("retired", None)),
        timer_factory=FakeTimer,
    )

    assert events == [("created", 35.0), ("started", True)]
    timer.callback()
    assert events[-1] == ("retired", None)


def test_zero_retirement_grace_is_immediate():
    events = []

    timer = schedule_retirement(0, lambda: events.append("retired"))

    assert timer is None
    assert events == ["retired"]


def test_metrics_render_aggregate_worker_state_and_bounded_labels():
    metrics = ProcessorMetrics()
    metrics.job_started("stream")
    metrics.pipeline_started("stream", 1.25)
    metrics.frame_processed("stream", 0.042, first_result_seconds=1.5)
    metrics.frame_processed("stream", 0.055)
    metrics.job_finished("stream", "completed")
    metrics.claim_rejected("execution_cell_mismatch")

    rendered = metrics.render(
        active_jobs=2,
        capacity=4,
        tier="gpu",
        cell="crusoe-use1",
        retiring=False,
        active_publishers={"whip": 1},
    )

    assert 'video_processor_info{cell="crusoe-use1",tier="gpu"} 1' in rendered
    assert "video_processor_busy 2" in rendered
    assert "video_processor_active_jobs 2" in rendered
    assert "video_processor_capacity 4" in rendered
    assert "video_processor_available_slots 2" in rendered
    assert 'video_processor_jobs_started_total{mode="stream"} 1' in rendered
    assert (
        'video_processor_jobs_finished_total{mode="stream",outcome="completed"} 1'
        in rendered
    )
    assert 'video_processor_frames_processed_total{mode="stream"} 2' in rendered
    assert 'video_processor_output_publishers{transport="whip"} 1' in rendered
    assert (
        'video_processor_claim_rejections_total{reason="execution_cell_mismatch"} 1'
        in rendered
    )
    assert (
        'video_processor_decode_to_result_latency_seconds_count{mode="stream"} 2'
        in rendered
    )
    assert (
        'video_processor_time_to_first_result_seconds_count{mode="stream"} 1'
        in rendered
    )


def test_metrics_normalize_untrusted_labels_and_never_include_job_identity():
    metrics = ProcessorMetrics()
    metrics.job_started('workspace/job-123\n"')
    metrics.job_finished("unexpected-mode", "unexpected-outcome")

    rendered = metrics.render(
        active_jobs=0,
        capacity=1,
        tier="unexpected-tier",
        cell="workspace/job-123",
        retiring=True,
    )

    assert "workspace" not in rendered
    assert "job-123" not in rendered
    assert "unexpected" not in rendered
    assert 'video_processor_info{cell="unknown",tier="unknown"} 1' in rendered
    assert 'video_processor_jobs_started_total{mode="unknown"} 1' in rendered
    assert (
        'video_processor_jobs_finished_total{mode="unknown",outcome="error"} 1'
        in rendered
    )
    assert "video_processor_retiring 1" in rendered


def test_metrics_use_a_bounded_legacy_cell_when_cell_is_not_configured():
    rendered = ProcessorMetrics().render(
        active_jobs=0,
        capacity=1,
        tier="cpu",
        retiring=False,
    )

    assert 'video_processor_info{cell="legacy",tier="cpu"} 1' in rendered
