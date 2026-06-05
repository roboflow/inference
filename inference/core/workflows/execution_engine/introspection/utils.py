import re
from typing import List

CLASS_NAME_WORD_PATTERN = re.compile(r"[A-Z](?:[a-z1-9]+|[A-Z]*(?=[A-Z]|$))")


def get_full_type_name(selected_type: type) -> str:
    t_class = selected_type.__name__
    t_module = selected_type.__module__
    if t_module == "builtins":
        return t_class.__qualname__
    return t_module + "." + selected_type.__qualname__


def build_human_friendly_block_name(
    fully_qualified_name: str, block_schema: dict = None
) -> str:
    if block_schema is not None:
        manual_title = block_schema.get("name")
        if manual_title is not None:
            return manual_title

    class_name = fully_qualified_name.split(".")[-1]
    if len(class_name) == 0:
        return f"Unrecognized Block Name: {fully_qualified_name}"
    words = _words_from_class_name(class_name=class_name)
    if len(words) > 0 and words[-1] == "Block":
        words.pop()
    if len(words) == 0:
        return f"Unrecognized Block Name: {fully_qualified_name}"
    return " ".join(words)


def _words_from_class_name(class_name: str) -> List[str]:
    words = []
    for segment in class_name.split("_"):
        if not segment:
            continue
        segment_words = _snake_cased_words_from_class_segment(class_segment=segment)
        if len(segment_words) == 0:
            segment_words = [segment]
        words.extend(segment_words)
    return words


def _snake_cased_words_from_class_segment(class_segment: str) -> List[str]:
    return CLASS_NAME_WORD_PATTERN.findall(class_segment)
