from typing import Optional, Callable, Any, Iterable
import torch
import optree

from lidra.data.utils import tree_transpose_level_one


def remove_empty_items_from_batch(batch):
    return [x for x in batch if x is not None]


def can_be_stacked(items: Iterable[Any]):
    for i, item in enumerate(items):
        if not isinstance(item, torch.Tensor):
            return False
        if i > 0:
            if (
                (shape != item.shape)
                or (dtype != item.dtype)
                or (device != item.device)
            ):
                return False
        else:
            shape = item.shape
            dtype = item.dtype
            device = item.device
    return True


def batch_leaf(leaf: Iterable[Any]):
    if can_be_stacked(leaf):
        return torch.stack(leaf, dim=0)
    return leaf


def unbatch_leaf(leaf: Iterable[Any]):
    if isinstance(leaf, torch.Tensor):
        leaf = torch.chunk(leaf, leaf.shape[0], dim=0)
        leaf = tuple(torch.squeeze(x, dim=0) for x in leaf)
    return leaf


def auto_collate(is_leaf: Optional[Callable] = None):

    def collate_fn(batch):
        batch = remove_empty_items_from_batch(batch)
        batch = tree_transpose_level_one(
            batch,
            check_children=False,
            map_fn=batch_leaf,
            is_leaf=is_leaf,
        )
        return batch

    return collate_fn


def auto_uncollate(is_leaf: Optional[Callable] = None):

    def uncollate_fn(batch):
        chunked_batch = optree.tree_map(
            unbatch_leaf,
            batch,
            is_leaf=is_leaf,
        )
        unbatched = tree_transpose_level_one(
            chunked_batch,
            check_children=False,
            is_leaf=None,
        )
        return unbatched

    return uncollate_fn
