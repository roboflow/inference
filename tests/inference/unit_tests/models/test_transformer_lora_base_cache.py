import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

LORA_BASE_URL = "https://example.com/base.tar.gz?signature=1"


def _model(transformers_module):
    model = transformers_module.LoRATransformerModel.__new__(
        transformers_module.LoRATransformerModel
    )
    model.api_key = "key"
    model.device_id = "device"

    return model


def _prepare(
    monkeypatch,
    tmp_path,
    transformers_module,
    *,
    offline=False,
    fail_first_download=False,
):
    cache_dir = tmp_path / "lora-bases" / "org" / "base" / "main"
    monkeypatch.setattr(transformers_module, "MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(transformers_module, "OFFLINE_MODE", offline)
    monkeypatch.setattr(
        transformers_module, "get_cache_dir", lambda model_id: str(cache_dir)
    )
    monkeypatch.setattr(
        transformers_module,
        "get_cache_file_path",
        lambda file, model_id: str(cache_dir / file),
    )
    api_calls = []
    monkeypatch.setattr(
        transformers_module,
        "get_roboflow_base_lora",
        lambda *args: api_calls.append(args) or {"weights": {"model": LORA_BASE_URL}},
    )

    download_calls = []

    def fake_stream_url_to_cache(url, filename, model_id):
        download_calls.append(url)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if fail_first_download and len(download_calls) == 1:
            raise RuntimeError("download failed")
        content = b"{}"
        member = tarfile.TarInfo(name="config.json")
        member.size = len(content)
        with tarfile.open(cache_dir / filename, "w:gz") as tar:
            tar.addfile(member, io.BytesIO(content))

    monkeypatch.setattr(
        transformers_module, "stream_url_to_cache", fake_stream_url_to_cache
    )

    return cache_dir, api_calls


def test_incomplete_lora_base_is_rebuilt(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    cache_dir.mkdir(parents=True)
    (cache_dir / "partial.bin").write_bytes(b"partial")

    result = _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert len(api_calls) == 1
    assert not (cache_dir / "partial.bin").exists()
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


def test_fresh_lora_base_is_downloaded_and_marked(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)

    _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert len(api_calls) == 1
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


def test_complete_lora_base_is_reused(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"{}")
    (cache_dir / ".complete").touch()

    _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert api_calls == []
    assert (cache_dir / "config.json").exists()


def test_offline_mode_trusts_existing_lora_base_without_marker(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers, offline=True)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"{}")

    result = _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert api_calls == []
    assert (cache_dir / "config.json").exists()
    assert not (cache_dir / ".complete").exists()


def test_offline_mode_miss_raises_before_touching_the_cache(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers, offline=True)

    with pytest.raises(transformers.ModelArtefactError):
        _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert api_calls == []
    assert not cache_dir.exists()
    assert not (tmp_path / "_file_locks").exists()


def test_failed_download_leaves_no_marker_and_next_call_retries(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(
        monkeypatch, tmp_path, transformers, fail_first_download=True
    )
    model = _model(transformers)

    with pytest.raises(RuntimeError):
        model.get_lora_base_from_roboflow("org/base", "main")

    assert not (cache_dir / ".complete").exists()

    result = model.get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert len(api_calls) == 2
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


def test_failed_extraction_leaves_no_marker_and_next_call_rebuilds(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    model = _model(transformers)
    original_extractall = tarfile.TarFile.extractall

    def _fail_extraction(self, path):
        (cache_dir / "partial.bin").write_bytes(b"partial")
        raise tarfile.ReadError("interrupted extraction")

    with monkeypatch.context() as patch:
        patch.setattr(tarfile.TarFile, "extractall", _fail_extraction)
        with pytest.raises(tarfile.ReadError, match="interrupted extraction"):
            model.get_lora_base_from_roboflow("org/base", "main")

    assert not (cache_dir / ".complete").exists()
    assert tarfile.TarFile.extractall is original_extractall

    model.get_lora_base_from_roboflow("org/base", "main")

    assert len(api_calls) == 2
    assert not (cache_dir / "partial.bin").exists()
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


def test_concurrent_loads_wait_for_complete_lora_base(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    download_started = Event()
    finish_download = Event()
    second_lock_attempted = Event()
    original_download = transformers.stream_url_to_cache
    original_lock = transformers.FileLock

    def _download(url, filename, model_id):
        cache_dir.mkdir(parents=True)
        download_started.set()
        assert finish_download.wait(timeout=10)
        original_download(url, filename, model_id)

    class ObservedFileLock(original_lock):
        def acquire(self, *args, **kwargs):
            if download_started.is_set():
                second_lock_attempted.set()
            return super().acquire(*args, **kwargs)

    monkeypatch.setattr(transformers, "stream_url_to_cache", _download)
    monkeypatch.setattr(transformers, "FileLock", ObservedFileLock)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            _model(transformers).get_lora_base_from_roboflow, "org/base", "main"
        )
        try:
            assert download_started.wait(timeout=10)
            second = executor.submit(
                _model(transformers).get_lora_base_from_roboflow, "org/base", "main"
            )
            assert second_lock_attempted.wait(timeout=10)
            assert not second.done()
        finally:
            finish_download.set()

        assert first.result(timeout=10) == str(cache_dir)
        assert second.result(timeout=10) == str(cache_dir)

    assert len(api_calls) == 1
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()
