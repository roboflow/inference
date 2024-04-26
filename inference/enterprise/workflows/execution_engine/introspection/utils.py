import re


def get_full_type_name(selected_type: type) -> str:
    t_class = selected_type.__name__
    t_module = selected_type.__module__
    if t_module == "builtins":
        return t_class.__qualname__
    return t_module + "." + selected_type.__qualname__


def build_human_friendly_block_name(fully_qualified_name: str) -> str:
    class_name = get_class_name_from_fully_qualified_name(
        fully_qualified_name=fully_qualified_name
    )
    return make_block_class_name_human_friendly(name=class_name)


def get_class_name_from_fully_qualified_name(fully_qualified_name: str) -> str:
    return fully_qualified_name.split(".")[-1]


def make_block_class_name_human_friendly(name: str) -> str:
    words = re.findall(r"[A-Z](?:[a-z1-9]+|[A-Z]*(?=[A-Z]|$))", name)
    if words[-1] == "Block":
        words.pop()
    return " ".join(words)
