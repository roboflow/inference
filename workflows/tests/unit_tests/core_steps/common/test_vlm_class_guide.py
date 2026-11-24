import pytest
from roboflow_workflows.core_steps.common.vlm_class_guide import (
    ClassGuide,
    class_guide_text,
    validate_class_guide,
    with_class_guide,
)

GUIDE = {
    "scene": "X-rays of one hand.",
    "classes": {
        "Ulna": "The narrower forearm bone.",
        "Radius": "The wider forearm bone.",
    },
}


def test_class_guide_text_lists_the_scene_then_the_classes_in_label_order() -> None:
    # when
    text = class_guide_text(ClassGuide(**GUIDE), labels=["Radius", "Ulna"])

    # then
    assert text == (
        "\n\nWhat the labels mean, from labeled examples of this dataset:\n"
        "X-rays of one hand.\n"
        "- Radius: The wider forearm bone.\n"
        "- Ulna: The narrower forearm bone."
    )


def test_class_guide_text_keeps_one_line_per_class_and_skips_empty_entries() -> None:
    # given
    guide = ClassGuide(
        scene="  ",
        classes={"Radius": "The wider\nforearm   bone. ", "Ulna": " \n "},
    )

    # when
    text = class_guide_text(guide, labels=["Radius", "Ulna", "MCP"])

    # then
    assert text == (
        "\n\nWhat the labels mean, from labeled examples of this dataset:\n"
        "- Radius: The wider forearm bone."
    )


@pytest.mark.parametrize(
    "guide",
    [None, ClassGuide(classes={}), ClassGuide(scene=" ", classes={"Ulna": ""})],
)
def test_class_guide_text_is_empty_without_text(guide) -> None:
    # when
    text = class_guide_text(guide, labels=["Radius", "Ulna"])

    # then
    assert text == ""


def test_validate_class_guide_parses_a_dict() -> None:
    # when
    guide = validate_class_guide(
        GUIDE, task_type="object-detection", labels=["Radius", "Ulna"]
    )

    # then
    assert guide == ClassGuide(**GUIDE)


def test_validate_class_guide_without_a_guide_is_none() -> None:
    # when
    guide = validate_class_guide(None, task_type="classification", labels=None)

    # then
    assert guide is None


def test_validate_class_guide_rejects_other_tasks() -> None:
    with pytest.raises(ValueError, match="only supported for the object-detection"):
        validate_class_guide(GUIDE, task_type="classification", labels=None)


def test_validate_class_guide_rejects_classes_the_block_does_not_have() -> None:
    with pytest.raises(ValueError, match=r"does not have: \['Ulna'\]"):
        validate_class_guide(GUIDE, task_type="object-detection", labels=["Radius"])


def test_validate_class_guide_accepts_any_class_when_labels_are_unknown() -> None:
    # when
    guide = validate_class_guide(GUIDE, task_type="object-detection", labels=None)

    # then
    assert guide == ClassGuide(**GUIDE)


@pytest.mark.parametrize(
    "value",
    [
        "a guide",
        {"scene": "X-rays."},
        {"classes": {"Ulna": 3}},
        {"classes": {}, "notes": "unknown key"},
    ],
)
def test_validate_class_guide_rejects_a_malformed_guide(value) -> None:
    with pytest.raises(ValueError, match="`class_guide` is not a valid guide"):
        validate_class_guide(value, task_type="object-detection", labels=None)


def test_with_class_guide_appends_to_the_last_text_part() -> None:
    # given
    parts = [
        {"text": "Reference image 1:"},
        {"inline_data": {"mime_type": "image/jpeg", "data": "ref"}},
        {"inline_data": {"mime_type": "image/jpeg", "data": "target"}},
        {"text": "Detect all objects."},
    ]

    # when
    combined = with_class_guide(parts, guide_text="\n\nGuide.")

    # then
    assert combined[:3] == parts[:3]
    assert combined[3] == {"text": "Detect all objects.\n\nGuide."}
    assert parts[3] == {"text": "Detect all objects."}


def test_with_class_guide_without_text_keeps_the_parts() -> None:
    # given
    parts = [{"text": "Detect all objects."}]

    # when
    combined = with_class_guide(parts, guide_text="")

    # then
    assert combined == parts
