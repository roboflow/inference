from pytorch3d.implicitron.dataset.dataset_base import FrameData

from lidra.data.collator import remove_empty_items_from_batch


def collate_fn(batch):
    batch = remove_empty_items_from_batch(batch)
    if len(batch) > 0:
        for x in batch:
            assert len(x) == 4
        return (
            FrameData.collate([x[0] for x in batch]),
            FrameData.collate([x[1] for x in batch]),
            [x[2] for x in batch],
            [x[3] for x in batch],
        )
    return None
