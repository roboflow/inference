import functools
from typing import Any, Callable, Union

from omegaconf import DictConfig, ListConfig, OmegaConf
from hydra.utils import instantiate

TargetType = Union[str, type, Callable[..., Any]]
ClassOrCallableType = Union[type, Callable[..., Any]]


def dump_config(config: DictConfig, path: str = "./config.yaml"):
    txt = OmegaConf.to_yaml(config, sort_keys=True)
    with open(path, "w") as f:
        f.write(txt)


def locate(path: str) -> Any:
    if path == "":
        raise ImportError("Empty path")

    import builtins
    from importlib import import_module

    parts = [part for part in path.split(".") if part]

    # load module part
    module = None
    for n in reversed(range(len(parts))):
        try:
            mod = ".".join(parts[:n])
            module = import_module(mod)
        except Exception as e:
            if n == 0:
                raise ImportError(f"Error loading module '{path}'") from e
            continue
        if module:
            break

    if module:
        obj = module
    else:
        obj = builtins

    # load object path in module
    for part in parts[n:]:
        mod = mod + "." + part
        if not hasattr(obj, part):
            try:
                import_module(mod)
            except Exception as e:
                raise ImportError(
                    f"Encountered error: `{e}` when loading module '{path}'"
                ) from e
        obj = getattr(obj, part)

    return obj


def full_instance_name(instance: Any) -> str:
    return full_class_name(instance.__class__)


def full_class_name(klass: Any) -> str:
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def ensure_is_subclass(child_class: type, parent_class: type) -> None:
    if not issubclass(child_class, parent_class):
        raise RuntimeError(
            f"class {full_class_name(child_class)} should be a subclass of {full_class_name(parent_class)}"
        )


def find_class_or_callable_from_target(
    target: TargetType,
) -> ClassOrCallableType:
    if isinstance(target, str):
        obj = locate(target)
    else:
        obj = target

    if (not isinstance(obj, type)) and (not callable(obj)):
        raise ValueError(f"Invalid type ({type(obj)}) found for {target}")

    return obj


def find_and_ensure_is_subclass(target: TargetType, type_: type) -> ClassOrCallableType:
    klass = find_class_or_callable_from_target(target)
    ensure_is_subclass(klass, type_)
    return klass


class StrictPartial:
    # remark : the `/` will handle the `path` argument name conflict (e.g. calling StrictPartial("a.b.c", ..., path="/a/b/c"))
    def __init__(self, path, /, *args, **kwargs):
        class_or_callable = find_class_or_callable_from_target(path)
        self._partial = functools.partial(class_or_callable, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._partial(*args, **kwargs)


class RecursivePartial:
    @staticmethod
    def replace_keys(config, key_mapping):
        def recurse(data):
            if isinstance(data, DictConfig):
                new_data = {
                    key_mapping[k] if k in key_mapping else k: recurse(v)
                    for k, v in data.items()
                }
                new_data = DictConfig(new_data)
            elif isinstance(data, ListConfig):
                new_data = ListConfig([recurse(item) for item in data])
            elif type(data) in {bool, str, int, float, type(None)}:
                new_data = data
            else:
                raise RuntimeError(f"unknow type found : {type(data)}")

            return new_data

        return recurse(config)

    def __init__(self, config):
        self.config = RecursivePartial.replace_keys(
            config, {"_rpartial_target_": "_target_"}
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return instantiate(self.config)


class Partial(StrictPartial):
    # remark : allow `path` argument to be exposed for easier use
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)


def subkey(mapping, key):
    return mapping[key]


def make_set(*args):
    return set(args)


def make_tuple(*args):
    return tuple(args)


def make_list_from_kwargs(**kwargs):
    # Filter out None/null values to avoid issues with callbacks
    return [v for v in kwargs.values() if v is not None]


def make_string(value):
    return str(value)


def make_dict(**kwargs):
    return dict(kwargs)


def get_item(data, key: str):
    return data[key]


def get_attr(data, key: str):
    return getattr(data, key)
