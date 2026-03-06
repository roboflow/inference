from typing import List, Dict, Any
from copy import deepcopy

import torch
from torch.optim import Optimizer
from loguru import logger

from lidra.test.util import OverwriteTensorEquality


class Composite(Optimizer):
    """Simple optimizer container acting as one single optimizer."""

    def __init__(self, *optimizers: List[Optimizer], check_consistency=True):
        super().__init__((torch.nn.Parameter(torch.zeros(1)),), {})

        # empty mock parameter (hack)
        self.param_groups.pop()
        assert len(self.param_groups) == 0

        self._optimizers = optimizers
        self._check_consistency = check_consistency

        # get params from provided optimizers
        for opt in self._optimizers:
            for pg in opt.param_groups:
                self.add_param_group(pg)

        self._do_check_consistency()

    def __repr__(self):
        return f"{self.__class__.__name__}(*{repr(self._optimizers)})"

    def zero_grad(self):
        for op in self._optimizers:
            op.zero_grad()
        self._do_check_consistency()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for op in self._optimizers:
            op.step(closure=None)

        self._do_check_consistency()

        return loss

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        self._do_check_consistency()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        i = 0
        for opt in self._optimizers:
            for idx2 in range(len(opt.param_groups)):
                opt.param_groups[idx2] = self.param_groups[i]
                i += 1

        self._do_check_consistency()

    def _do_check_consistency(self):
        if self._check_consistency:
            self_ids = set()
            opts_ids = set()
            for pg in self.param_groups:
                self_ids |= set(id(p) for p in pg["params"])

            for opt in self._optimizers:
                for pg in opt.param_groups:
                    opts_ids |= set(id(p) for p in pg["params"])

            assert self_ids == opts_ids

    def state_dict(self):
        """
        Return the state of the composite optimizer as a dict.

        Aggregates states from all constituent optimizers and maps them
        to the composite optimizer's parameter indexing scheme.

        Returns:
            dict: Dictionary containing 'state' and 'param_groups'
        """
        # Structure follows PyTorch's standard: {'state': {}, 'param_groups': []}
        composite_state = {"state": {}, "param_groups": deepcopy(self.param_groups)}

        try:
            # Build mapping from parameter object ID to composite parameter index
            param_id_to_composite_idx = self._build_param_mapping()

            # Aggregate states from individual optimizers
            for optimizer in self._optimizers:
                opt_state_dict = optimizer.state_dict()

                # Build reverse mapping for this optimizer: param index -> parameter object
                opt_param_idx_to_obj = {}
                param_idx = 0
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        opt_param_idx_to_obj[param_idx] = param
                        param_idx += 1

                # Map each parameter's state to composite indexing
                for opt_param_idx, param_state in opt_state_dict["state"].items():
                    if opt_param_idx in opt_param_idx_to_obj:
                        param_obj = opt_param_idx_to_obj[opt_param_idx]
                        composite_idx = param_id_to_composite_idx.get(id(param_obj))
                        if composite_idx is not None:
                            composite_state["state"][composite_idx] = deepcopy(
                                param_state
                            )

            logger.debug(
                f"CompositeOptimizer state_dict: collected {len(composite_state['state'])} parameter states"
            )
            return composite_state

        except Exception as e:
            logger.warning(f"Error building CompositeOptimizer state_dict: {e}")
            logger.warning("Returning state_dict with empty state")
            return composite_state

    def load_state_dict(self, state_dict):
        """
        Load the state of the composite optimizer from a state dict.

        Distributes the state to constituent optimizers based on parameter ownership.

        Args:
            state_dict (dict): State dictionary containing 'state' and 'param_groups'
        """
        try:
            # Validation
            if not isinstance(state_dict, dict):
                raise TypeError(f"Expected dict, got {type(state_dict)}")

            # Load state if present and not empty
            if "state" in state_dict and state_dict["state"]:
                logger.debug(
                    f"Loading CompositeOptimizer state with {len(state_dict['state'])} parameter states"
                )

                # Build reverse mapping: composite idx -> (optimizer, local_idx)
                composite_to_opt_mapping = self._build_reverse_param_mapping()

                # Prepare state dicts for individual optimizers
                optimizer_states = {}
                for i, optimizer in enumerate(self._optimizers):
                    optimizer_states[i] = {
                        "state": {},
                        "param_groups": deepcopy(
                            optimizer.param_groups
                        ),  # Keep current param_groups
                    }

                # Distribute states to individual optimizers
                for composite_idx, param_state in state_dict["state"].items():
                    composite_idx = int(composite_idx)  # Ensure it's an int
                    if composite_idx in composite_to_opt_mapping:
                        opt_idx, local_param_idx = composite_to_opt_mapping[
                            composite_idx
                        ]
                        optimizer_states[opt_idx]["state"][local_param_idx] = deepcopy(
                            param_state
                        )

                # Apply loaded states to individual optimizers
                for i, optimizer in enumerate(self._optimizers):
                    if optimizer_states[i]["state"]:
                        optimizer.load_state_dict(optimizer_states[i])
                        logger.debug(
                            f"Loaded {len(optimizer_states[i]['state'])} states into optimizer {i}"
                        )

            # Update param_groups AFTER loading states to preserve parameter references
            if "param_groups" in state_dict:
                # Only update the settings (lr, weight_decay, etc.) not the parameter lists
                for i, new_group in enumerate(state_dict["param_groups"]):
                    if i < len(self.param_groups):
                        # Preserve the original parameter list but update settings
                        original_params = self.param_groups[i]["params"]
                        self.param_groups[i] = deepcopy(new_group)
                        self.param_groups[i]["params"] = original_params

                # Update individual optimizers' param_groups to match
                self._distribute_param_groups_to_optimizers()
            else:
                logger.debug("No optimizer state to load (empty or missing)")

        except Exception as e:
            logger.warning(f"Failed to load CompositeOptimizer state: {e}")
            logger.warning("Continuing with fresh optimizer state")

    def _build_param_mapping(self):
        """Build mapping from parameter ID to composite parameter index."""
        param_id_to_idx = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group["params"]):
                # Calculate global parameter index across all param groups
                global_idx = (
                    sum(len(g["params"]) for g in self.param_groups[:group_idx])
                    + param_idx
                )
                param_id_to_idx[id(param)] = global_idx
        return param_id_to_idx

    def _build_reverse_param_mapping(self):
        """Build mapping from composite index to (optimizer_index, local_param_index)."""
        composite_to_opt = {}

        # Track which composite parameter index we're at
        composite_idx = 0

        # For each optimizer, map its parameters to composite indices
        for opt_idx, optimizer in enumerate(self._optimizers):
            local_param_idx = 0
            for opt_group in optimizer.param_groups:
                for param in opt_group["params"]:
                    # Map this local parameter index to the current composite index
                    composite_to_opt[composite_idx] = (opt_idx, local_param_idx)
                    composite_idx += 1
                    local_param_idx += 1

        return composite_to_opt

    def _distribute_param_groups_to_optimizers(self):
        """Update individual optimizers' param_groups from composite param_groups."""
        param_group_idx = 0
        for optimizer in self._optimizers:
            for opt_group_idx in range(len(optimizer.param_groups)):
                if param_group_idx < len(self.param_groups):
                    # Copy relevant parameters (lr, weight_decay, etc.) but keep original params list
                    original_params = optimizer.param_groups[opt_group_idx]["params"]
                    new_group = deepcopy(self.param_groups[param_group_idx])
                    new_group["params"] = original_params
                    optimizer.param_groups[opt_group_idx] = new_group
                    param_group_idx += 1

        # Only check consistency if checking is enabled
        if self._check_consistency:
            self._do_check_consistency()

    def _do_check_equality(self, other):
        # TODO(Pierre) : investigate, the line below doesn't work because it gets
        # filled in with a "differentiable" field after "__setstate__" is called
        # assert self.defaults == other.defaults
        with OverwriteTensorEquality(torch, lambda t0, t1: id(t0) == id(t1)):
            assert self.param_groups == other.param_groups
            assert self.state == other.state
