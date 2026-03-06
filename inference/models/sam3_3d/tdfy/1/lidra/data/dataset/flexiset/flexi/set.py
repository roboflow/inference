from typing import Union, Sequence, Dict

from lidra.data.dataset.flexiset.flexi.loader import Loader as FlexiLoader
from lidra.data.dataset.flexiset.flexi.solver import (
    solve_loaders_ordering,
    solve_minimal_transforms,
)

# loader = extract new information from existing information
# transform = change existing information


class Set:
    def __init__(
        self,
        inputs: Union[str, Sequence[str]],
        loaders: Dict[str, FlexiLoader],
        outputs: Union[str, Sequence[str]],
        transforms: Union[str, Sequence[str]] = (),
    ):
        super().__init__()

        self._compiled = False
        self._all_loaders = loaders
        self._all_transforms = transforms
        self._loaders = None
        self._transforms = None

        self.inputs = inputs
        self.outputs = outputs

    def _compile(self):
        if not self._compiled:
            # get minimal set of loaders
            self._loaders = solve_loaders_ordering(
                self._inputs,
                self._all_loaders,
                self._outputs,
            )
            # get minimal set of transforms
            self._transforms = solve_minimal_transforms(
                self._all_transforms,
                self._outputs,
            )

            self._compiled = True

    @staticmethod
    def _make_sure_is_set(arg: Union[str, Sequence[str]]):
        return {arg} if isinstance(arg, str) else set(arg)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = Set._make_sure_is_set(inputs)
        self._compiled = False

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = Set._make_sure_is_set(outputs)
        self._compiled = False

    def _check_inputs(self, inputs):
        if set(inputs) != self._inputs:
            raise ValueError(
                f"inputs do not match the expected inputs. "
                f"expected {self._inputs} but got {set(inputs)}"
            )

    def __call__(self, **inputs):
        self._check_inputs(inputs)
        self._compile()

        # gather inputs
        item = {**inputs}

        # load data
        for loader in self._loaders:
            item = loader(item)

        # transform data
        for transform in self._transforms:
            item = transform(item)

        # filter out unecessary keys
        item = {key: val for key, val in item.items() if key in self.outputs}

        return item
