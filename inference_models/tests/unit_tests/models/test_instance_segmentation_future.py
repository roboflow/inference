from inference_models.models.base.instance_segmentation import _DirectInferenceFuture


class _FakeEvent:
    def __init__(self, query_result: bool) -> None:
        self.query_result = query_result
        self.query_calls = 0

    def query(self) -> bool:
        self.query_calls += 1
        return self.query_result


class _FakeInstanceSegmentationModel:
    def __init__(self) -> None:
        self.calls = []
        self.result = ["detections"]

    def post_process(self, raw, meta, **kwargs):
        self.calls.append((raw, meta, dict(kwargs)))
        return self.result


def test_direct_inference_future_result_runs_postprocess_once() -> None:
    model = _FakeInstanceSegmentationModel()
    future = _DirectInferenceFuture(
        model=model,
        raw="raw-output",
        meta="metadata",
        evt=None,
        kwargs={"confidence": 0.4},
    )

    first_result = future.result()
    second_result = future.result()

    assert first_result is model.result
    assert second_result is model.result
    assert model.calls == [
        ("raw-output", "metadata", {"confidence": 0.4}),
    ]
    assert future.done() is True


def test_direct_inference_future_submit_gpu_work_is_idempotent() -> None:
    model = _FakeInstanceSegmentationModel()
    future = _DirectInferenceFuture(
        model=model,
        raw="raw-output",
        meta="initial-metadata",
        evt=None,
        kwargs={"mask_format": "rle"},
    )

    future.submit_gpu_work(meta="submitted-metadata")
    future.submit_gpu_work(meta="ignored-metadata")
    result = future.result()

    assert result is model.result
    assert future.preprocess_metadata == "submitted-metadata"
    assert model.calls == [
        ("raw-output", "submitted-metadata", {"mask_format": "rle"}),
    ]


def test_direct_inference_future_done_queries_event_until_cached() -> None:
    model = _FakeInstanceSegmentationModel()
    event = _FakeEvent(query_result=False)
    future = _DirectInferenceFuture(
        model=model,
        raw="raw-output",
        meta="metadata",
        evt=event,
        kwargs={},
    )

    assert future.done() is False
    assert event.query_calls == 1

    future.result()

    assert future.done() is True
    assert event.query_calls == 1
