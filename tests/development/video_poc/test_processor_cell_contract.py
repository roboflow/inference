from pathlib import Path

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)


def test_claim_validates_placement_before_any_execution_side_effect():
    source = (PROCESSOR_DIR / "processor.py").read_text()
    claim = source[
        source.index("    def try_claim(self):") : source.index(
            "    def _fresh_claim_holdoff"
        )
    ]

    validation = claim.index("validate_job_placement(job, self.cell)")
    assert validation < claim.index('job.pop("processorAccessToken", None)')
    assert validation < claim.index("self.security.register_job")
    assert validation < claim.index("run_type(self, job)")
    assert validation < claim.index("self.pod.detach_from_pool")
    assert (
        "self.report_job_failure"
        not in claim[validation : claim.index("access_token =")]
    )
    assert "self.retiring = True" in claim[validation : claim.index("access_token =")]


def test_platform_calls_carry_the_immutable_worker_cell_assertion():
    source = (PROCESSOR_DIR / "processor.py").read_text()

    assert 'self.cell = validate_cell_id(os.getenv("VIDEO_PROC_CELL"))' in source
    assert source.count("platform_identity()") >= 5
    assert '"cell": self.cell' in source
    assert "cell=worker.cell" in source
