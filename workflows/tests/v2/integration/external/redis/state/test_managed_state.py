"""Public ``ManagedState`` semantics on an owned Redis server (opt-in external suite).

The cases live in ``tests/v2/support/state/managed_state_contract.py``; the
in-memory run is ``unit/engine/state/test_managed_state.py``.
"""

from typing import Optional

import pytest
from roboflow_workflows.v2.engine.state import ManagedState

from tests.v2.support.state.managed_state_contract import (  # noqa: F401  collected here
    test_at_selects_a_member_of_a_batch_call,
    test_compare_and_set_equality_is_canonical,
    test_compare_and_set_with_missing,
    test_concurrent_compare_and_set_has_one_winner,
    test_concurrent_first_use_seeds_before_any_operation,
    test_concurrent_increments_are_exact,
    test_declared_defaults_are_stored_initial_values,
    test_global_and_source_scopes_are_isolated,
    test_incr_counts_from_zero_and_accepts_negative_amounts,
    test_incr_of_a_missing_key_returns_exactly_the_amount,
    test_incr_on_non_integer_raises_type_error_and_keeps_value,
    test_incr_overflow_raises_and_keeps_value,
    test_incr_returns_exactly_the_stored_value,
    test_initialize_once_shares_bookkeeping_with_defaults,
    test_invalid_amount_is_rejected_before_the_backend,
    test_invalid_declared_defaults_are_rejected,
    test_missing_differs_from_null,
    test_namespaces_are_isolated_with_unusual_names,
    test_new_service_on_the_same_namespace_seeds_only_absent_defaults,
    test_rejected_values_change_nothing,
    test_same_backend_and_namespace_share_state,
    test_set_only_if_absent_and_delete,
    test_source_defaults_seed_through_the_current_call,
    test_source_scope_rejects_every_batch_delivering_call,
    test_source_scope_rejects_missing_sources_and_contexts,
    test_source_scope_resolves_the_current_call,
    test_values_are_copied,
    test_views_of_one_service_initialize_each_key_once,
)


@pytest.fixture(params=["redis"])
def make_state(request, namespace):
    backends = []

    def factory(namespace_override: Optional[str] = None) -> ManagedState:
        from roboflow_workflows.v2.engine.state.redis import RedisStateBackend

        server = request.getfixturevalue("redis_server")
        backend = RedisStateBackend(server.url)
        backends.append(backend)
        state = ManagedState(backend, namespace=namespace_override or namespace)

        return state

    yield factory

    for backend in backends:
        backend.close()
