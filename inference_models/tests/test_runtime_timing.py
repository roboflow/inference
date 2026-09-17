"""CPU fake-event tests for diagnostic lifecycle, bounds, and thread isolation."""

import importlib.util
import threading
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "runtime_timing",
    Path(__file__).parents[1] / "inference_models/models/common/runtime_timing.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Event:
    ready = True

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        assert self.stream == end.stream
        return 2.5


class Image:
    shape = (3, 1080, 1920)


class TimingTest(unittest.TestCase):
    def sample(self, timer, success=True):
        timer.begin(Image())
        for stage in ("preprocess", "trt", "postprocess"):
            timer.mark(stage, stage)
            timer.mark(stage, stage, end=True)
        timer.finish(success)

    def test_sampling_and_shapes(self):
        timer = module.RuntimeTiming(Event, sample_every=2)
        self.sample(timer)
        self.assertEqual(timer.snapshot()["completed_samples"], 0)
        self.sample(timer)
        record = timer.snapshot()["records"][0]
        self.assertEqual(record["input_shapes"], [[3, 1080, 1920]])
        self.assertEqual(
            record["cuda_stream_ms"],
            dict.fromkeys(("preprocess", "trt", "postprocess"), 2.5),
        )
        self.assertGreaterEqual(record["cpu_wall_ms"]["total"], 0)

    def test_pending_bounded_and_nonblocking(self):
        class Pending(Event):
            ready = False

        timer = module.RuntimeTiming(Pending, sample_every=1, capacity=2)
        for _ in range(4):
            self.sample(timer)
        self.assertEqual(timer.snapshot()["pending_samples"], 2)
        self.assertEqual(timer.snapshot()["dropped_samples"], 2)
        Pending.ready = True
        timer.collect()
        self.assertEqual(timer.snapshot()["completed_samples"], 2)

    def test_failed_sample_cleared(self):
        timer = module.RuntimeTiming(Event, sample_every=1)
        self.sample(timer, success=False)
        timer.mark("preprocess", "standalone")
        self.assertIsNone(timer.local.active)
        self.assertEqual(timer.snapshot()["completed_samples"], 0)

    def test_threads_and_record_bound(self):
        timer = module.RuntimeTiming(Event, sample_every=1, capacity=3)
        threads = [
            threading.Thread(target=lambda: [self.sample(timer) for _ in range(30)])
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        result = timer.snapshot()
        self.assertEqual(result["completed_samples"], 120)
        self.assertEqual(len(result["records"]), 3)


if __name__ == "__main__":
    unittest.main()
