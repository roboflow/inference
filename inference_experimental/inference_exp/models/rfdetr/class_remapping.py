from typing import Dict, List, Set, Tuple, Union

from inference_exp.models.common.roboflow.model_packages import ClassNameRemoval


def prepare_class_remapping(
    class_names: List[str],
    class_names_operations: List[
        Union[ClassNameRemoval]
    ],  # be ready for different elements of union type
) -> Tuple[List[str], Dict[int, int]]:
    removed_classes = {
        o.class_name for o in class_names_operations if isinstance(o, ClassNameRemoval)
    }
    class_idx_remapping = {}
    removed_class_ids = set()
    result_classes = []
    for class_id, class_name in enumerate(class_names):
        if class_name in removed_classes:
            removed_class_ids.add(class_id)
            continue
        class_idx_remapping[class_id] = class_id - len(removed_class_ids)
        result_classes.append(class_name)
    return result_classes, class_idx_remapping
