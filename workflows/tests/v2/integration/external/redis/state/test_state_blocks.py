"""Native state blocks on an owned Redis server (opt-in external suite).

The cases live in ``tests/v2/support/state/state_blocks_contract.py``; the
in-memory run is ``unit/engine/state/test_state_blocks.py``.
"""

import pytest

from tests.v2.support.state.state_blocks_contract import (  # noqa: F401  collected here
    _state,
    test_compare_and_set_block_can_delete,
    test_compare_and_set_block_resolves_selectors_at_run_time,
    test_non_portable_value_fails_the_step,
    test_session_borrows_caller_state_and_seeds_each_default_once,
    test_source_scope_uses_the_input_source,
    test_source_scope_without_a_source_fails_the_step,
    test_state_blocks_chain_global_operations,
)


@pytest.fixture(params=["redis"])
def state(request):
    managed_state = _state(request)
    yield managed_state
    managed_state.backend.close()
