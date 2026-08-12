import json
import os
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from development.mmp_staging_benchmark.capability_probe import shared_memory_report
from development.mmp_staging_benchmark.run_concurrent_clients import (
    ClientRecorder,
    ClientSpec,
    _metrics_evidence,
    client_name,
    jain_fairness,
    latency_summary,
    load_spec,
    routing_key,
    validate_staging_url,
)

DiskUsage = namedtuple("DiskUsage", "total used free")


class StagingTargetTests(unittest.TestCase):
    def test_accepts_local_private_kubernetes_and_staging(self) -> None:
        accepted = [
            "http://127.0.0.1:8000",
            "http://10.1.2.3:8000",
            "http://mmp.video-proc.svc:8000",
            "https://mmp-benchmark.roboflow.one",
            "https://inference-staging.example.net",
        ]
        for target in accepted:
            with self.subTest(target=target):
                self.assertEqual(validate_staging_url(target), target)

    def test_rejects_known_or_arbitrary_public_hosts(self) -> None:
        for target in [
            "https://api.roboflow.com",
            "https://video-processors.crusoe.roboflow.com",
            "https://example.com",
        ]:
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "refusing non-staging"):
                    validate_staging_url(target)


class SpecTests(unittest.TestCase):
    def test_same_tenant_can_drive_multiple_clients_with_client_ids(self) -> None:
        raw = {
            "server_url": "http://127.0.0.1:8000",
            "duration_s": 10,
            "clients": [
                {
                    "client_id": "tenant-a-det",
                    "tenant_id": "tenant-a",
                    "api_key_env": "TENANT_A_KEY",
                    "model_id": "detector",
                },
                {
                    "client_id": "tenant-a-seg",
                    "tenant_id": "tenant-a",
                    "api_key_env": "TENANT_A_KEY",
                    "model_id": "segmenter",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "spec.json"
            path.write_text(json.dumps(raw))
            spec = load_spec(path)
        self.assertEqual(
            [client_name(c) for c in spec.clients],
            ["tenant-a-det", "tenant-a-seg"],
        )

    def test_duplicate_effective_client_ids_are_rejected(self) -> None:
        raw = {
            "server_url": "http://127.0.0.1:8000",
            "clients": [
                {"tenant_id": "a", "api_key_env": "A", "model_id": "m"},
                {"tenant_id": "a", "api_key_env": "A", "model_id": "m2"},
            ],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "spec.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "client_id"):
                load_spec(path)

    def test_instance_creates_distinct_mmp_routing_key(self) -> None:
        shared = ClientSpec("tenant-a", "KEY", "model")
        isolated = ClientSpec("tenant-a", "KEY", "model", instance="tenant-a")
        self.assertEqual(routing_key(shared), "model")
        self.assertEqual(routing_key(isolated), "model:tenant-a")


class StatisticsTests(unittest.TestCase):
    def test_latency_summary_interpolates_quantiles(self) -> None:
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertEqual(summary["max"], 4.0)

    def test_jain_fairness(self) -> None:
        self.assertEqual(jain_fairness([10, 10, 10]), 1.0)
        self.assertAlmostEqual(jain_fairness([10, 0]), 0.5)
        self.assertIsNone(jain_fairness([0, 0]))

    def test_recorder_excludes_warmup_from_capacity_counts(self) -> None:
        client = ClientSpec("tenant", "KEY", "model")
        recorder = ClientRecorder(client, max_samples=10, seed=1)
        recorder.record(
            offset_s=0.0,
            measured_offset_s=-1.0,
            latency_ms=100,
            status=200,
            error=None,
            measured=False,
        )
        recorder.record(
            offset_s=1.0,
            measured_offset_s=0.0,
            latency_ms=20,
            status=200,
            error=None,
            measured=True,
        )
        report = recorder.report(measured_duration_s=2)
        self.assertEqual(report["requests"], 1)
        self.assertEqual(report["delivered_fps"], 0.5)
        self.assertEqual(report["first_success_latency_ms"], 100)

    def test_metrics_evidence_records_batch_and_vram_deltas(self) -> None:
        samples = [
            {
                "offset_s": 0,
                "status": 200,
                "metrics": {
                    "gpus": [
                        {
                            "utilization_pct": 10,
                            "memory_used_mb": 1000,
                            "power_w": 50,
                        }
                    ],
                    "per_model_gpu_mb": {"m": 900},
                    "mmp_free_slots": 10,
                    "mmp_pending": 0,
                    "mmp_rejects_pool_full": 1,
                    "mmp_models": {"m": {"inference_count": 2, "batch_count": 1}},
                },
            },
            {
                "offset_s": 1,
                "status": 200,
                "metrics": {
                    "gpus": [
                        {
                            "utilization_pct": 90,
                            "memory_used_mb": 1500,
                            "power_w": 150,
                        }
                    ],
                    "per_model_gpu_mb": {"m": 1400},
                    "mmp_free_slots": 2,
                    "mmp_pending": 8,
                    "mmp_rejects_pool_full": 3,
                    "mmp_models": {
                        "m": {
                            "inference_count": 22,
                            "batch_count": 6,
                            "avg_batch_size": 4,
                            "worker_pid": 42,
                        }
                    },
                },
            },
        ]
        evidence = _metrics_evidence(samples)
        self.assertEqual(evidence["gpu_utilization_pct"]["max"], 90)
        self.assertEqual(evidence["per_model_peak_vram_mb"]["m"], 1400)
        self.assertEqual(evidence["mmp_pool_full_rejects_delta"], 2)
        self.assertEqual(evidence["model_deltas"]["m"]["inference_count_delta"], 20)
        self.assertEqual(evidence["model_deltas"]["m"]["batch_count_delta"], 5)


class CapabilityProbeTests(unittest.TestCase):
    def test_shared_memory_geometry_includes_reserve(self) -> None:
        with patch.dict(
            os.environ,
            {
                "INFERENCE_N_SLOTS": "2",
                "INFERENCE_INPUT_MB": "1",
                "MMP_SHM_RESERVE_FACTOR": "1.25",
            },
            clear=False,
        ), patch("pathlib.Path.exists", return_value=True), patch(
            "os.access", return_value=True
        ), patch(
            "shutil.disk_usage",
            return_value=DiskUsage(total=4_000_000, used=0, free=4_000_000),
        ):
            report = shared_memory_report("/fake-shm")
        self.assertEqual(report["required_bytes"], 2 * (1024 * 1024 + 64))
        self.assertEqual(
            report["recommended_bytes"], int(report["required_bytes"] * 1.25)
        )
        self.assertTrue(report["satisfies_recommended"])


if __name__ == "__main__":
    unittest.main()
