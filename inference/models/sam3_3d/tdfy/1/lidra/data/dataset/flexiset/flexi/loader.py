from typing import Dict, Any, Union, Sequence, Mapping
from loguru import logger

from lidra.data.dataset.flexiset.loaders.base import Base as BaseLoader
from lidra.data.dataset.flexiset.transforms.base import Base as BaseTransform
from lidra.data.dataset.flexiset.flexi.transform import Transform as FlexiTransform
from lidra.data.dataset.flexiset.transforms.path.repath import Repath


class _Dependencies:
    def __init__(self, loader, input_mapping):
        self.loader = loader
        self.input_mapping = input_mapping

    def _remap_dependencies(self, dependencies):
        return {
            (self.input_mapping[dep] if dep in self.input_mapping else dep)
            for dep in dependencies
        }

    @property
    def required(self):
        dependencies = set(self.loader.args.required_parameters)
        dependencies -= set(self.loader.default_loaders)
        return self._remap_dependencies(dependencies)

    @property
    def optional(self):
        dependencies = set(self.loader.args.optional_parameters)
        dependencies |= set(self.loader.default_loaders)
        if self.loader.args.args_parameter is not None:
            dependencies.add(self.loader.args.args_parameter)
        if self.loader.args.kwargs_parameter is not None:
            dependencies.add(self.loader.args.kwargs_parameter)
        return self._remap_dependencies(dependencies)

    def dependencies_from_available_nodes(self, nodes):
        dependencies = set()
        # push the loader outputs to process next
        for dep in self.required:
            dependencies.add(dep)
        for dep in self.optional:
            if dep not in nodes:
                if dep in self.loader.default_loaders:
                    logger.warning(
                        f"optional dependency '{dep}' from loader {self.loader} not found, will use default loader instead"
                    )
                    default_dependencies = _Dependencies(
                        self.loader.default_loaders[dep],
                        self.input_mapping,
                    )
                    dependencies |= (
                        default_dependencies.dependencies_from_available_nodes(nodes)
                    )
                else:
                    logger.warning(
                        f"optional dependency '{dep}' from loader {self.loader} not found, will use default value instead"
                    )
            else:
                dependencies.add(dep)
        return dependencies


class Loader:
    def __init__(
        self,
        outputs: Union[str, Sequence[str]],
        loader: BaseLoader,
        input_mapping: Mapping[str, str] = {},
        subpath: str = "",
    ):
        # TODO(Pierre) : check loader signature by binding outputs / dependencies
        self.loader = loader
        self.input_mapping = dict(input_mapping)
        self.dependencies = _Dependencies(loader, self.input_mapping)
        self.outputs = (outputs,) if isinstance(outputs, str) else tuple(outputs)
        self._pre_transforms = []
        self._post_transforms = []

        if len(subpath) > 0:
            self.with_subpath(subpath)

    def dependencies_from_available_nodes(self, nodes):
        return self.dependencies.dependencies_from_available_nodes(nodes)

    def with_pre_transforms(self, *transforms):
        pre_transforms = list(transforms)
        pre_transforms.extend(self._pre_transforms)
        self._pre_transforms = pre_transforms
        return self

    def with_post_transforms(self, *transforms):
        self._post_transforms.extend(transforms)
        return self

    with_transform = with_post_transforms

    def with_subpath(self, subpath: str):
        return self.with_pre_transforms(FlexiTransform(("path",), Repath(subpath)))

    def __call__(self, item: Dict[str, Any]):
        kwargs = dict(item)
        for key, item_key in self.input_mapping.items():
            if not item_key in kwargs:
                continue  # will use default value instead
            kwargs[key] = kwargs.pop(item_key)

        for transform in self._pre_transforms:
            kwargs = transform(kwargs)

        result = self.loader.load(**kwargs)

        if len(self.outputs) == 1:
            result = (result,)
        if len(result) != len(self.outputs):
            raise ValueError(
                f"Number of outputs {len(self.outputs)} does not match the "
                f"length of the returned result {len(result)}"
            )

        result = {key: result[i] for i, key in enumerate(self.outputs)}
        new_item = {**item, **result}
        return new_item

    def __repr__(self):
        outputs = ",".join(self.outputs)
        required_dependencies = ",".join(self.dependencies.required)
        optional_dependencies = ",".join(self.dependencies.optional)
        fn = self.loader.__class__.__name__
        return f"{outputs} = {fn}({required_dependencies}, ?, {optional_dependencies})"
