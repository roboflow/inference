# Model registry inventories

`registry_inventory.json` is a reviewed snapshot of current registration keys and
their preferred class paths in all adapter/proxy configurations. When adding,
removing or changing a model registration or its feature flag, run from the repo root:

```sh
python tests/inference/unit_tests/models/generate_registry_inventory.py
python -m pytest tests/inference/unit_tests/models/test_lazy_model_registry.py tests/inference/unit_tests/models/test_registry_inventory_generator.py
```

Commit the generated JSON with the registration change and review its key, path
and flag changes. Tests compare with the committed snapshot; they never update it
automatically. The generator reads current source and discovers its enabled flags;
it does not need an eager implementation, Git history, or optional model libraries.
Its flags describe inventory coverage, not production defaults. OWLv2 and SAM3D
are excluded from that default inventory configuration; dedicated tests cover them.

Disabled-flag tests additionally require each model flag to control registry keys
when adapters are enabled, so removing a guard cannot pass by changing both sides
of a live-source comparison. The generator's small metadata fakes do not simulate
runtime imports or fallback: the availability, failure, concurrency and real-import
tests cover those separately.

`model_export_inventory.json` retains the earlier public class exports as a
backward-compatibility contract. Adding a registry key does not require changing
those existing exports.
