"""Unit tests for registry_defaults — class-vs-MRO skip logic on _register_from_config."""

from inference_model_manager import registry_defaults
from inference_model_manager.registry import ModelRegistry


def test_subclass_override_registers_after_base(monkeypatch):
    """Subclass config must register even when a base class with same task is already registered.

    Regression: previously _register_from_config used get_entry_for_class (MRO-walking)
    to decide whether to skip. After Base.infer was registered, Sub.infer would be
    silently skipped, and Sub instances would inherit Base's validator instead of
    their own (e.g. open-vocabulary detection missing the `classes` requirement).
    """

    def base_validator(kwargs):
        return {"validator": "base", **kwargs}

    def sub_validator(kwargs):
        return {"validator": "sub", **kwargs}

    def fake_serializer(out, model):
        return {}

    fake_configs = {
        "FakeBase": [
            ("infer", "infer", True, {}, "base_v", "fake_ser", "fake-base-v1"),
        ],
        "FakeSub": [
            ("infer", "infer", True, {}, "sub_v", "fake_ser", "fake-sub-v1"),
        ],
    }
    fake_validators = {"base_v": base_validator, "sub_v": sub_validator}
    fake_serializers = {"fake_ser": fake_serializer}

    test_registry = ModelRegistry()
    monkeypatch.setattr(registry_defaults, "registry", test_registry)
    monkeypatch.setattr(registry_defaults, "_TASK_CONFIGS", fake_configs)
    monkeypatch.setattr(
        registry_defaults,
        "_resolve_validator",
        lambda name: fake_validators[name],
    )
    monkeypatch.setattr(
        registry_defaults,
        "_resolve_serializer",
        lambda name: fake_serializers[name],
    )

    class FakeBase:
        pass

    class FakeSub(FakeBase):
        pass

    registry_defaults._register_from_config(FakeBase)
    registry_defaults._register_from_config(FakeSub)

    base_entry = test_registry.get_entry(FakeBase(), "infer")
    sub_entry = test_registry.get_entry(FakeSub(), "infer")

    assert base_entry is not None
    assert sub_entry is not None
    assert base_entry.response_type == "fake-base-v1"
    assert sub_entry.response_type == "fake-sub-v1"
    assert sub_entry.validator({}) == {"validator": "sub"}


def test_register_from_config_idempotent_for_same_class(monkeypatch):
    """Calling _register_from_config twice on same class must not duplicate or overwrite."""

    def validator(kwargs):
        return kwargs

    def serializer(out, model):
        return {}

    fake_configs = {
        "FakeBase": [
            ("infer", "infer", True, {}, "v", "s", "fake-v1"),
        ],
    }

    test_registry = ModelRegistry()
    monkeypatch.setattr(registry_defaults, "registry", test_registry)
    monkeypatch.setattr(registry_defaults, "_TASK_CONFIGS", fake_configs)
    monkeypatch.setattr(registry_defaults, "_resolve_validator", lambda _: validator)
    monkeypatch.setattr(registry_defaults, "_resolve_serializer", lambda _: serializer)

    class FakeBase:
        pass

    registry_defaults._register_from_config(FakeBase)
    first_entry = test_registry.get_entry(FakeBase(), "infer")

    registry_defaults._register_from_config(FakeBase)
    second_entry = test_registry.get_entry(FakeBase(), "infer")

    assert first_entry is second_entry
    assert test_registry.registered_tasks(FakeBase) == ["infer"]
