from collections import namedtuple
from typing import List, Tuple, Union

import torch

from inference_models.models.common.roboflow.model_packages import ClassNameRemoval

ClassesReMapping = namedtuple(
    "ClassesReMapping", ["remaining_class_ids", "class_mapping"]
)


def prepare_class_remapping(
    class_names: List[str],
    class_names_operations: List[
        Union[ClassNameRemoval]
    ],  # be ready for different elements of union type
    device: torch.device,
) -> Tuple[List[str], ClassesReMapping]:
    removed_classes = {
        o.class_name for o in class_names_operations if isinstance(o, ClassNameRemoval)
    }
    removed_class_ids = set()
    remaining_class_ids = []
    result_classes = []
    class_mapping = []
    for class_id, class_name in enumerate(class_names):
        if class_name in removed_classes:
            removed_class_ids.add(class_id)
            class_mapping.append(-1)
            continue
        remaining_class_ids.append(class_id)
        class_mapping.append(class_id - len(removed_class_ids))
        result_classes.append(class_name)
    classes_re_mapping = ClassesReMapping(
        remaining_class_ids=torch.tensor(
            remaining_class_ids, dtype=torch.int64, device=device
        ),
        class_mapping=torch.tensor(class_mapping, dtype=torch.int64, device=device),
    )
    return result_classes, classes_re_mapping
