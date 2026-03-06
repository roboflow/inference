from typing import Dict, Any, Union, Sequence

from lidra.data.dataset.flexiset.transforms.base import Base as BaseTransform


class Transform:
    def __init__(
        self,
        inputs: Union[str, Sequence[str]],
        transform: BaseTransform,
    ):
        # TODO(Pierre) : check loader signature by bind outputs / dependencies
        self.transform = transform
        self.inputs = (inputs,) if isinstance(inputs, str) else tuple(inputs)

    def __call__(self, item: Dict[str, Any]):
        args = tuple(item[key] for key in self.inputs)

        result = self.transform.transform(*args)

        result = {key: result[i] for i, key in enumerate(self.inputs)}
        new_item = {**item, **result}
        return new_item

    def __repr__(self):
        outputs = ",".join(self.outputs)
        required_dependencies = ",".join(self.dependencies.required)
        optional_dependencies = ",".join(self.dependencies.optional)
        return (
            f"{self.__class__.__name__}("
            f"{outputs} = fn({required_dependencies}, ?, {optional_dependencies})"
        )
