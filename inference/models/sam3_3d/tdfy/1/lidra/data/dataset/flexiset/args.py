import inspect
from typing import Sequence, Mapping

## COMMENT(Pierre) Keep this comment below for reference on how the signature extraction works

# def example_function(POSITIONAL_OR_KEYWORD):
#     pass
# def example_function(POSITIONAL_OR_KEYWORD, *, KEYWORD_ONLY):
#     pass
# def example_function(POSITIONAL_ONLY, /, POSITIONAL_OR_KEYWORD):
#     pass
# def example_function(*VAR_POSITIONAL):
#     pass
# def example_function(**VAR_KEYWORDS):
#     pass
# def example_function(POSITIONAL_ONLY, /, POSITIONAL_OR_KEYWORD, *, KEYWORD_ONLY):
#     pass
# def example_function(POSITIONAL_OR_KEYWORD, POSITIONAL_OR_KEYWORD_ALSO=None):
#     pass
# def example_function(POSITIONAL_ONLY, *VAR_POSITIONAL, KEYWORD_ONLY):
#     pass


class Args:
    def __init__(self, fn):
        self._extract_load_signature(fn)

    @property
    def required_parameters(self):
        return self._required_parameters

    @property
    def optional_parameters(self):
        return self._optional_parameters

    @property
    def args_parameter(self):
        return self._args_parameter

    @property
    def kwargs_parameter(self):
        return self._kwargs_parameter

    def _extract_load_signature(self, fn):
        self._signature = inspect.signature(fn)

        self._required_parameters = set()
        self._optional_parameters = set()
        self._args_parameter = None
        self._kwargs_parameter = None

        for param in self._signature.parameters.values():
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise ValueError(
                    f"POSITIONAL_ONLY parameters are not (yet) supported in the "
                    f"_load method of {self.__class__.__name__}. "
                    f"Please use POSITIONAL_OR_KEYWORD or KEYWORD_ONLY."
                )
            elif param.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                if param.default is inspect.Parameter.empty:
                    self._required_parameters.add(param.name)
                else:
                    self._optional_parameters.add(param.name)
            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                self._args_parameter = param.name
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                self._kwargs_parameter = param.name

    def bind(self, all_kwargs):
        reduced_kwargs = {}

        # check if required parameters are present in all_kwargs
        for key in self._required_parameters:
            if key not in all_kwargs:
                raise ValueError(f"Missing required parameter '{key}'.")

        required_kwargs = {key: all_kwargs[key] for key in self._required_parameters}
        optional_kwargs = {
            key: all_kwargs[key]
            for key in self._optional_parameters
            if key in all_kwargs
        }
        reduced_kwargs = {**required_kwargs, **optional_kwargs}

        # merge kwargs
        if self._kwargs_parameter is not None:
            kwargs = all_kwargs[self._kwargs_parameter]
            if not isinstance(kwargs, Mapping):
                raise ValueError(
                    f"Expected '{self._kwargs_parameter}' to be a mapping, but got "
                    f"{type(kwargs).__name__}."
                )
            kwargs = dict(kwargs)
            for key in kwargs:
                if key in reduced_kwargs:
                    raise ValueError(
                        f"Conflict in parameter '{key}': it is defined in both "
                        f"'{self._kwargs_parameter}' and as a required/optional parameter."
                    )
                reduced_kwargs[key] = kwargs[key]

        # prepare args
        if self._args_parameter is not None:
            args = all_kwargs[self._args_parameter]
            if not isinstance(args, Sequence):
                raise ValueError(
                    f"Expected '{self._args_parameter}' to be a sequence, but got "
                    f"{type(args).__name__}."
                )
            args = tuple(args)
        else:
            args = ()

        return args, reduced_kwargs
