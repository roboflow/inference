import re


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
    words = re.findall(r"[A-Z](?:[a-z1-9]+|[A-Z]*(?=[A-Z]|$))", class_name)
    if words[-1] == "Block":
        words.pop()
    return " ".join(words)
