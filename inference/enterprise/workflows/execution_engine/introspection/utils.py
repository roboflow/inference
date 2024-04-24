import re


def get_full_type_name(t: type) -> str:
    t_class = t.__name__
    t_module = t.__module__
    if t_module == "builtins":
        return t_class.__qualname__
    return t_module + "." + t.__qualname__


def build_human_friendly_block_name(fully_qualified_name: str) -> str:
    class_name = get_class_name_from_fully_qualified_name(
        fully_qualified_name=fully_qualified_name
    )
    return block_class_name_to_block_title(name=class_name)


def get_class_name_from_fully_qualified_name(fully_qualified_name: str) -> str:
    return fully_qualified_name.split(".")[-1]


def block_class_name_to_block_title(name: str) -> str:
    words = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", name)
    if words[-1] == "Block":
        words.pop()
    return " ".join(words)
