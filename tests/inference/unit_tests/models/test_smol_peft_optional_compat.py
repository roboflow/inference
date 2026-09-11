"""Dependency-free unit tests of the exact file-compatibility helper AST.

Installed PEFT construction is separately tested in the candidate runtime.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
import typing
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / "inference/models/transformers/transformers.py"
SMOL = ROOT / "inference/models/smolvlm/smolvlm.py"
NULL_KEYS = ("lora_ga_config", "use_bdlora")


def load_helper(offline):
    tree = ast.parse(HELPER.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_compatible_adapter_config"
    )
    scope = {"json": json, "os": os, "OFFLINE_MODE": offline,
             "Iterable": typing.Iterable, "Dict": typing.Dict, "Any": typing.Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(HELPER), "exec"), scope)
    return scope["load_compatible_adapter_config"]


class SmolOptionalPeftCompatibility(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="smol-peft-unit-")
        self.path = Path(self.directory.name) / "adapter_config.json"

    def tearDown(self):
        self.directory.cleanup()

    def run_helper(self, config, offline=True, unsupported=()):
        raw = json.dumps(config, sort_keys=True).encode()
        self.path.write_bytes(raw)
        result = load_helper(offline)(
            str(self.path), unsupported_keys=unsupported,
            unsupported_null_keys=NULL_KEYS,
        )
        return result, raw

    def test_explicit_null_defaults_removed_in_memory_offline(self):
        original = {"r": 8, "lora_ga_config": None, "use_bdlora": None}
        result, raw = self.run_helper(original)
        self.assertEqual(result, {"r": 8})
        self.assertEqual(self.path.read_bytes(), raw)

    def test_explicit_null_defaults_never_rewrite_online_artifact(self):
        result, raw = self.run_helper({"r": 8, "lora_ga_config": None,
                                       "use_bdlora": None}, offline=False)
        self.assertEqual(result, {"r": 8})
        self.assertEqual(self.path.read_bytes(), raw)

    def test_non_null_options_fail_closed_without_any_write(self):
        for key in NULL_KEYS:
            for value in (False, True, 0, "", {}, {"enabled": True}):
                with self.subTest(key=key, value=value):
                    config = {"r": 8, "eva_config": {"rho": 2}, key: value}
                    raw = json.dumps(config).encode(); self.path.write_bytes(raw)
                    with self.assertRaisesRegex(ValueError, key):
                        load_helper(False)(str(self.path), unsupported_keys=["eva_config"],
                                           unsupported_null_keys=NULL_KEYS)
                    self.assertEqual(self.path.read_bytes(), raw)

    def test_unknown_fields_remain_for_native_peft_rejection(self):
        result, raw = self.run_helper({"r": 8, "future_option": None})
        self.assertEqual(result, {"r": 8, "future_option": None})
        self.assertEqual(self.path.read_bytes(), raw)

    def test_absent_options_leave_config_and_file_unchanged(self):
        result, raw = self.run_helper({"r": 8})
        self.assertEqual(result, {"r": 8})
        self.assertEqual(self.path.read_bytes(), raw)

    def test_existing_online_compatibility_persistence_is_preserved(self):
        result, _ = self.run_helper({"r": 8, "eva_config": {},
                                    "lora_ga_config": None, "use_bdlora": None},
                                   offline=False, unsupported=["eva_config"])
        self.assertEqual(result, {"r": 8})
        self.assertEqual(json.loads(self.path.read_text()),
                         {"r": 8, "lora_ga_config": None, "use_bdlora": None})

    def test_smol_explicitly_selects_only_the_two_proven_null_fields(self):
        calls = [node for node in ast.walk(ast.parse(SMOL.read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "load_compatible_adapter_config"]
        self.assertEqual(len(calls), 1)
        option = next(k.value for k in calls[0].keywords
                      if k.arg == "unsupported_null_keys")
        self.assertEqual(ast.literal_eval(option), list(NULL_KEYS))


class SmolResolvedProcessorCache(unittest.TestCase):
    def initialize(self, model_id, offline, processor_error=None, base_error=None):
        tree = ast.parse(SMOL.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LoRASmolVLM")
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_model")
        resolved = "/model-cache/~resolved-native-base-0123456789"
        self.model = Mock()
        self.peft = Mock()
        self.peft.from_pretrained.return_value.eval.return_value.to.return_value = self.model
        self.base = Mock()
        self.processor = Mock()
        self.processor.from_pretrained.side_effect = processor_error
        self.subject = SimpleNamespace(cache_dir="/model-cache/~adapter", get_lora_base_from_roboflow=Mock(return_value=resolved), transformers_class=self.base, processor_class=self.processor)
        self.subject.get_lora_base_from_roboflow.side_effect = base_error
        scope = {"os": os, "torch": SimpleNamespace(bfloat16="native-bfloat16"),
                 "load_compatible_adapter_config": Mock(return_value={}),
                 "LoraConfig": Mock(return_value=SimpleNamespace(base_model_name_or_path=model_id, revision="main")),
                 "MODEL_CACHE_DIR": "/model-cache", "OFFLINE_MODE": offline,
                 "DEVICE": "cuda:0", "is_flash_attn_2_available": lambda: False,
                 "remove_extracted_archive_if_online": Mock(), "PeftModel": self.peft}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SMOL), "exec"), scope)
        scope["initialize_model"](self.subject)
        return resolved

    def test_both_sizes_use_resolved_weight_directory_online_and_offline(self):
        for model_id in ("smolvlm2", "smolvlm2/smolvlm-256m"):
            for offline in (False, True):
                with self.subTest(model=model_id, offline=offline):
                    resolved = self.initialize(model_id, offline)
                    self.processor.from_pretrained.assert_called_once_with(resolved, local_files_only=offline)
                    self.subject.get_lora_base_from_roboflow.assert_called_once_with(model_id, "main")
                    self.base.from_pretrained.assert_called_once_with(resolved, revision=None, device_map="cuda:0", cache_dir=resolved, token=None, attn_implementation="eager", local_files_only=offline)
                    self.peft.from_pretrained.return_value.eval.return_value.to.assert_called_once_with("native-bfloat16")
                    self.model.merge_and_unload.assert_called_once_with()

    def test_processor_failure_propagates_without_alternate_path_or_retry(self):
        with self.assertRaisesRegex(OSError, "invalid processor"):
            self.initialize("smolvlm2", True, processor_error=OSError("invalid processor"))
        self.assertEqual(self.processor.from_pretrained.call_count, 1)

    def test_native_base_resolution_failure_never_falls_back_to_raw_path(self):
        with self.assertRaisesRegex(OSError, "missing native archive"):
            self.initialize("smolvlm2/smolvlm-256m", True, base_error=OSError("missing native archive"))
        self.base.from_pretrained.assert_not_called()
        self.processor.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
