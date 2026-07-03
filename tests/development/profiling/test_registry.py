import textwrap

from development.profiling.registry import registered_target_names, resolve_target


def test_builtin_target_lookup_resolves_smoke_target():
    target = resolve_target(name="smoke-tensor")

    assert target.name == "smoke-tensor"
    assert "smoke-tensor" in registered_target_names()


def test_generated_target_import_path_resolves_selected_file(tmp_path):
    target_file = tmp_path / "target.py"
    target_file.write_text(
        textwrap.dedent(
            """
            import torch


            class GeneratedTarget:
                name = "generated"

                def prepare(self, record, *, device):
                    prepared = torch.tensor([1.0], device=device)

                    return prepared

                def run(self, prepared):
                    output = prepared + 1

                    return output

                def validate(self, output):
                    assert output.item() == 2.0

                def summarize(self, output):
                    summary = {"value": float(output.item())}

                    return summary


            target = GeneratedTarget()
            """
        ),
        encoding="utf-8",
    )

    target = resolve_target(
        name="generated",
        import_path=f"{target_file}:target",
    )

    assert target.name == "generated"
    assert target.summarize(target.run(target.prepare(None, device="cpu"))) == {
        "value": 2.0
    }
