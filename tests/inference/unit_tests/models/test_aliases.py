from inference.models.aliases import resolve_roboflow_model_alias


def test_resolve_roboflow_model_alias_when_registry_hit_should_happen() -> None:
    # when
    result = resolve_roboflow_model_alias(model_id="yolov8m-1280")

    # then
    assert result == "coco/11"


def test_resolve_roboflow_model_alias_when_registry_hit_should_not_happen() -> None:
    # when
    result = resolve_roboflow_model_alias(model_id="my_project/3")

    # then
    assert result == "my_project/3"
