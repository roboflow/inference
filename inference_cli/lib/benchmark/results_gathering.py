from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

STATISTICS_FORMAT = """
avg: {average_inference_latency_ms}ms\t| rps: {requests_per_second}\t| p75: {p75_inference_latency_ms}ms\t| p90: {p90_inference_latency_ms}\t| %err: {error_rate}\t| {error_status_codes}
""".strip()


@dataclass(frozen=True)
class InferenceStatistics:
    inferences_made: int
    images_processed: int
    average_inference_latency_ms: float
    std_inference_latency_ms: float
    average_inference_latency_per_image_ms: float
    p50_inference_latency_ms: float
    p75_inference_latency_ms: float
    p90_inference_latency_ms: float
    p95_inference_latency_ms: float
    p99_inference_latency_ms: float
    requests_per_second: float
    images_per_second: float
    error_rate: float
    error_status_codes: Dict[str, int]

    def to_string(self) -> str:
        return STATISTICS_FORMAT.format(
            average_inference_latency_ms=self.average_inference_latency_ms,
            requests_per_second=self.requests_per_second,
            p50_inference_latency_ms=self.p50_inference_latency_ms,
            p75_inference_latency_ms=self.p75_inference_latency_ms,
            p90_inference_latency_ms=self.p90_inference_latency_ms,
            error_rate=self.error_rate,
            error_status_codes=self.error_status_codes,
        )


class ResultsCollector:

    def __init__(self):
        self._benchmark_start: Optional[datetime] = None
        self._inference_details: List[Tuple[datetime, int, float]] = []
        self._benchmark_end: Optional[datetime] = None
        self._errors: List[Tuple[datetime, int, str]] = []

    def start_benchmark(self) -> None:
        if self._benchmark_start is None:
            self._benchmark_start = datetime.now()

    def register_inference_duration(self, batch_size: int, duration: float) -> None:
        self._inference_details.append((datetime.now(), batch_size, duration))

    def register_error(self, batch_size: int, status_code: str) -> None:
        self._errors.append((datetime.now(), batch_size, status_code))

    def stop_benchmark(self) -> None:
        if self._benchmark_end is None:
            self._benchmark_end = datetime.now()

    def has_benchmark_finished(self) -> bool:
        return self._benchmark_end is not None

    def get_statistics(
        self, window: Optional[int] = None
    ) -> Optional[InferenceStatistics]:
        if self._benchmark_start is None or len(self._inference_details) < 1:
            return None
        end_time = (
            self._benchmark_end if self._benchmark_end is not None else datetime.now()
        )
        stats = copy(
            self._inference_details
        )  # to have it stable in multi-threading env
        errors = copy(self._errors)
        if window is not None:
            stats = stats[-window:]
        latencies = [s[2] for s in stats]
        inferences_made = len(stats)
        images_processed = sum(s[1] for s in stats)
        average_inference_latency_ms = round(np.average(latencies) * 1000, 1)
        std_inference_latency_ms = round(np.std(latencies) * 1000, 1)
        average_inference_latency_per_image_ms = round(
            average_inference_latency_ms * inferences_made / images_processed, 2
        )
        p50_inference_latency_ms = round(np.percentile(latencies, 50) * 1000, 1)
        p75_inference_latency_ms = round(np.percentile(latencies, 75) * 1000, 1)
        p90_inference_latency_ms = round(np.percentile(latencies, 90) * 1000, 1)
        p95_inference_latency_ms = round(np.percentile(latencies, 95) * 1000, 1)
        p99_inference_latency_ms = round(np.percentile(latencies, 99) * 1000, 1)
        start = (
            self._benchmark_start
            if window is None or len(stats) < window
            else stats[0][0]
        )

        error_status_codes = defaultdict(int)
        errors_number = 0
        for e in errors:
            if e[0] < start:
                continue
            error_status_codes[e[2]] += 1
            errors_number += 1

        error_rate = round(errors_number / inferences_made * 100, 2)
        duration = (end_time - start).total_seconds()
        requests_per_second = round(inferences_made / duration, 1)
        images_per_second = round(images_processed / duration, 1)
        return InferenceStatistics(
            inferences_made=inferences_made,
            images_processed=images_processed,
            average_inference_latency_ms=average_inference_latency_ms,
            std_inference_latency_ms=std_inference_latency_ms,
            average_inference_latency_per_image_ms=average_inference_latency_per_image_ms,
            p50_inference_latency_ms=p50_inference_latency_ms,
            p75_inference_latency_ms=p75_inference_latency_ms,
            p90_inference_latency_ms=p90_inference_latency_ms,
            p95_inference_latency_ms=p95_inference_latency_ms,
            p99_inference_latency_ms=p99_inference_latency_ms,
            requests_per_second=requests_per_second,
            images_per_second=images_per_second,
            error_rate=error_rate,
            error_status_codes=", ".join(
                f"{exc}: {count}" for exc, count in error_status_codes.items()
            ),
        )
