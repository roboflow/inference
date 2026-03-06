import os
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from lidra.config.utils import RecursivePartial
from lidra.utils.device import get_auto_device


def register(resolver_fn):
    OmegaConf.register_new_resolver(f"rv.{resolver_fn.__name__}", resolver_fn)
    return resolver_fn


# TODO(Pierre) : Add recursive option ?
@register
def partial(data, *, _node_, _parent_, _root_):
    if "_target_" not in data:
        raise RuntimeError(
            "`partial` resolver cannot resolve a node missing the `_target_` field"
        )

    new_data = {
        "_target_": "lidra.config.utils.StrictPartial",
        "_args_": (data["_target_"],) + data.get("_args_", ()),
        **{k: v for k, v in data.items() if k not in {"_target_", "_args_"}},
    }

    return DictConfig(new_data, parent=_parent_)


@register
def rpartial(data, *, _node_, _parent_, _root_):
    if "_target_" not in data:
        raise RuntimeError(
            "`partial` resolver cannot resolve a node missing the `_target_` field"
        )
    new_data = {
        "_target_": "lidra.config.utils.RecursivePartial",
        "config": RecursivePartial.replace_keys(
            data, {"_target_": "_rpartial_target_"}
        ),
    }

    return DictConfig(new_data, parent=_parent_)


@register
def locate(path: str, *, _node_, _parent_, _root_):
    return DictConfig({"_target_": "lidra.config.utils.locate", "path": path})


@register
def getitem(data, key, *, _node_, _parent_, _root_):
    return DictConfig(
        {
            "_target_": "lidra.config.utils.get_item",
            "data": data,
            "key": key,
        },
        parent=_parent_,
    )


@register
def getattr(data, key, *, _node_, _parent_, _root_):
    return DictConfig(
        {
            "_target_": "lidra.config.utils.get_attr",
            "data": data,
            "key": key,
        },
        parent=_parent_,
    )


@register
def freeze(data, *, _node_, _parent_, _root_):
    return DictConfig(
        {
            "_target_": "lidra.config.model.freeze",
            "model": data,
        },
        parent=_parent_,
    )


@register
def config(data, *, _node_, _parent_, _root_):
    return DictConfig(OmegaConf.load(data), parent=_parent_)


@register
def config_resolved(data, *, _node_, _parent_, _root_):
    loaded = OmegaConf.load(data)
    OmegaConf.resolve(loaded)  # Resolve in-place
    return DictConfig(loaded, parent=_parent_)


@register
def merge(*data, _node_, _parent_, _root_):
    if len(data) > 0:
        with open_dict(data[0]):
            data = [{} if d is None else d for d in data]
            return DictConfig(OmegaConf.merge(*data), parent=_parent_)
    return None


@register
def select_merge(base, overrides, *, _node_, _parent_, _root_):
    """Merge base with overrides, handling null values gracefully.

    If base is null/None, returns overrides (or null if both are null).
    If overrides is null/None, returns base.
    Otherwise merges base with overrides, with overrides taking precedence.
    """
    if base is None:
        return overrides  # Could be None or a dict
    elif overrides is None:
        return base
    else:
        # Both are non-null, merge them
        with open_dict(base):
            return DictConfig(OmegaConf.merge(base, overrides), parent=_parent_)


@register
def config_node(config_path, node_path, *, _node_, _parent_, _root_):
    """Loads a specific node from a config file.

    Args:
        config_path: Path to the config file
        node_path: Dot-separated path to the node within the config

    Returns:
        A DictConfig containing the specified node from the config, or the string value
        if the node is a string
    """
    # Load the full config
    full_config = OmegaConf.load(config_path)

    # Navigate to the specified node
    node = full_config
    if node_path:
        for key in node_path.split("."):
            if key in node:
                node = node[key]
            else:
                raise ValueError(
                    f"Key '{key}' not found in node path '{node_path}' in config '{config_path}'"
                )

    # Return string directly if node is a string
    if isinstance(node, str):
        return node

    return DictConfig(node, parent=_parent_)


@register
def config_node_resolved(config_path, node_path, *, _node_, _parent_, _root_):
    """Loads a specific node from a config file and resolves all interpolations.

    Args:
        config_path: Path to the config file
        node_path: Dot-separated path to the node within the config

    Returns:
        A DictConfig containing the resolved node from the config
    """
    # Load the full config
    full_config = OmegaConf.load(config_path)

    # Resolve all interpolations in the full config
    OmegaConf.resolve(full_config)

    # Navigate to the specified node
    node = full_config
    if node_path:
        for key in node_path.split("."):
            if key in node:
                node = node[key]
            else:
                raise ValueError(
                    f"Key '{key}' not found in node path '{node_path}' in config '{config_path}'"
                )

    # Return string directly if node is a string
    if isinstance(node, str):
        return node

    return DictConfig(node, parent=_parent_)


@register
def auto_device(device=None, *, _node_=None, _parent_=None, _root_=None):
    return get_auto_device(device)
