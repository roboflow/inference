from inference.core.workflows.execution_engine.v1.introspection.types_discovery import (
    discover_kinds_schemas,
    discover_kinds_typing_hints,
)


def test_discover_kinds_typing_hints() -> None:
    # when
    result = discover_kinds_typing_hints(kinds_names={"image", "float", "dictionary"})

    # then
    assert result == {"image": "dict", "float": "float", "dictionary": "dict"}


def test_discover_kinds_schemas() -> None:
    # when
    result = discover_kinds_schemas(
        kinds_names={"image", "float", "dictionary", "classification_prediction"}
    )

    # then
    assert set(result.keys()) == {
        "image",
        "classification_prediction",
    }, "Only image and object_detection_prediction kinds are expected to define schemas"
    assert isinstance(result["image"], dict), "Expected to be dict with schemas"
    assert isinstance(
        result["classification_prediction"], list
    ), "Expected to be list with schemas union"
