import sys
from pathlib import Path

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from processor_metrics import ProcessorMetrics  # noqa: E402


def test_processor_image_includes_metrics_module():
    dockerfile = (PROCESSOR_DIR / "Dockerfile").read_text()

    assert "COPY processor_metrics.py /app/processor_metrics.py" in dockerfile


def test_metrics_render_aggregate_worker_state_and_bounded_labels():
    metrics = ProcessorMetrics()
    metrics.job_started("stream")
    metrics.pipeline_started("stream", 1.25)
    metrics.frame_processed("stream", 0.042, first_result_seconds=1.5)
    metrics.frame_processed("stream", 0.055)
    metrics.job_finished("stream", "completed")

    rendered = metrics.render(
        active_jobs=2,
        capacity=4,
        tier="gpu",
        retiring=False,
        active_publishers={"whip": 1},
    )

    assert 'video_processor_info{tier="gpu"} 1' in rendered
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
        retiring=True,
    )

    assert "workspace" not in rendered
    assert "job-123" not in rendered
    assert "unexpected" not in rendered
    assert 'video_processor_info{tier="unknown"} 1' in rendered
    assert 'video_processor_jobs_started_total{mode="unknown"} 1' in rendered
    assert (
        'video_processor_jobs_finished_total{mode="unknown",outcome="error"} 1'
        in rendered
    )
    assert "video_processor_retiring 1" in rendered
