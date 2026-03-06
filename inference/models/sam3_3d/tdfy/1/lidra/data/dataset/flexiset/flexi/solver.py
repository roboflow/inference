from typing import Tuple, Sequence
from collections import namedtuple
from loguru import logger
import inspect

from lidra.data.dataset.flexiset.flexi.loader import Loader as FlexiLoader

EXPLORE_STATE = 0
REGISTER_STATE = 1

SolverState = namedtuple("SolverState", ("type", "key", "path"))


def _map_output_key_to_loader(loaders: Sequence[FlexiLoader]):
    output_to_loader = {}
    for loader in loaders:
        for output in loader.outputs:
            if output in output_to_loader:
                raise RuntimeError(
                    f"Output '{output}' is already used by another loader."
                )
            output_to_loader[output] = loader
    return output_to_loader


def _check_outputs(output_to_loader, outputs, inputs):
    for output in outputs:
        if (output not in output_to_loader) and (output not in inputs):
            raise RuntimeError(f"Output '{output}' is not provided by any loader.")


# TODO(Pierre) should clean a bit more, good enough for now since interface should be stable
def _solve(output_to_loader, outputs, inputs):
    available_nodes = set(inputs) | set(output_to_loader)
    ordered_loaders = []
    visited_loaders = set()
    in_branch_loaders = set()
    loader_to_dynamic_dependencies = {}

    nodes_to_process = list(
        SolverState(EXPLORE_STATE, output, ()) for output in outputs
    )
    # for output in outputs:
    while len(nodes_to_process) > 0:
        state = nodes_to_process.pop(-1)
        if state.key in inputs:
            continue
        if not state.key in output_to_loader:
            raise RuntimeError(
                f"Required key '{state.key}' was not found. Make sure it is provided as an input or loader."
            )
        loader: FlexiLoader = output_to_loader[state.key]
        if loader in visited_loaders:
            pass
        elif state.type is REGISTER_STATE:
            visited_loaders.add(loader)
            ordered_loaders.append(loader)
            in_branch_loaders.remove(loader)
        elif state.type is EXPLORE_STATE:
            if loader in in_branch_loaders:
                cycle = "->".join(state.path + (state.key,))
                raise RuntimeError(f"cycles detected in loaders dependencies : {cycle}")
            # explore dependencies first
            in_branch_loaders.add(loader)
            # finally register the current node
            nodes_to_process.append(SolverState(REGISTER_STATE, state.key, None))

            dependencies = loader.dependencies_from_available_nodes(available_nodes)

            loader_to_dynamic_dependencies[loader] = dependencies

            for dep in dependencies:
                if dep in inputs:
                    continue
                nodes_to_process.append(
                    SolverState(EXPLORE_STATE, dep, state.path + (state.key,))
                )

    return ordered_loaders, loader_to_dynamic_dependencies


def _print_loaders(loaders, loader_to_dynamic_dependencies):
    logger.debug("loaders ordering")
    for i, loader in enumerate(loaders):
        fn = loader.loader.__class__.__name__
        args = ",".join(loader_to_dynamic_dependencies[loader])
        outputs = ",".join(loader.outputs)
        func_call = f"{fn}({args})"
        source_file = inspect.getsourcefile(type(loader.loader))
        _, source_line = inspect.getsourcelines(type(loader.loader))
        source = f"{source_file}:{source_line}"
        logger.debug(f"#{i:04d} : {outputs} <- {func_call} [{source}]")


def solve_loaders_ordering(
    inputs: Tuple[str],
    loaders: FlexiLoader,
    outputs: Tuple[str],
):
    output_to_loader = _map_output_key_to_loader(loaders)
    _check_outputs(output_to_loader, outputs, inputs)
    ordered_loaders, loader_to_dynamic_dependencies = _solve(
        output_to_loader,
        outputs,
        inputs,
    )
    _print_loaders(ordered_loaders, loader_to_dynamic_dependencies)
    return ordered_loaders


def solve_minimal_transforms(all_transforms, outputs):
    outputs = set(outputs)
    transforms_to_keep = set()
    for transform in all_transforms:
        for key in transform.inputs:
            if key in outputs:
                transforms_to_keep.add(transform)
    return [trans for trans in all_transforms if trans in transforms_to_keep]
