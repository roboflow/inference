"""Optional bounded CUDA stream interval measurements, without synchronization."""

import threading
import time
from collections import deque


class RuntimeTiming:
    def __init__(self, event_factory, sample_every=10, capacity=2048):
        self.event_factory = event_factory
        self.sample_every = sample_every
        self.capacity = capacity
        self.local = threading.local()
        self.lock = threading.Lock()
        self.records = deque(maxlen=capacity)
        self.pending = deque()
        self.completed = 0
        self.dropped = 0

    def begin(self, images):
        self.collect()
        count = getattr(self.local, "count", 0) + 1
        self.local.count = count
        self.local.active = None
        if count % self.sample_every == 0:
            images = images if isinstance(images, list) else [images]
            self.local.active = {
                "input_shapes": [list(image.shape) for image in images],
                "events": {},
                "cpu_wall_ms": {},
                "started": time.perf_counter(),
            }

    def mark(self, stage, stream, end=False):
        active = getattr(self.local, "active", None)
        if active is None:
            return
        event = self.event_factory()
        event.record(stream)
        now = time.perf_counter()
        if end:
            start, _, started = active["events"][stage]
            active["events"][stage] = (start, event, started)
            active["cpu_wall_ms"][stage] = (now - started) * 1000
        else:
            active["events"][stage] = (event, None, now)

    def finish(self, success):
        active = getattr(self.local, "active", None)
        self.local.active = None
        if active is None or not success:
            return
        active["completed_at"] = time.time()
        active["cpu_wall_ms"]["total"] = (
            time.perf_counter() - active.pop("started")
        ) * 1000
        with self.lock:
            if len(self.pending) >= self.capacity:
                self.pending.popleft()
                self.dropped += 1
            self.pending.append(active)
        self.collect()

    def collect(self):
        with self.lock:
            waiting = deque()
            for item in self.pending:
                events = item["events"]
                if len(events) != 3 or any(
                    end is None for _, end, _ in events.values()
                ):
                    self.dropped += 1
                    continue
                if not all(end.query() for _, end, _ in events.values()):
                    waiting.append(item)
                    continue
                item["cuda_stream_ms"] = {
                    stage: start.elapsed_time(end)
                    for stage, (start, end, _) in events.items()
                }
                del item["events"]
                self.records.append(item)
                self.completed += 1
            self.pending = waiting

    def snapshot(self):
        # Collection happens in inference threads with their established CUDA context.
        with self.lock:
            return {
                "enabled": True,
                "sample_every": self.sample_every,
                "completed_samples": self.completed,
                "pending_samples": len(self.pending),
                "dropped_samples": self.dropped,
                "records": list(self.records),
            }
