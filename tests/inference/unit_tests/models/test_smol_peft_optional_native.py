"""Native PEFT constructor contract; fixture is the unchanged, SHA-bound real adapter.

Candidate image builds explicitly set SMOL_ADAPTER_CONTRACT_FIXTURE and
SMOL_EXPECTED_PEFT_VERSION. Ordinary unit runs skip this external-fixture suite.
No base model, GPU, weights, network or storage credentials are used.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FIXTURE = os.environ.get("SMOL_ADAPTER_CONTRACT_FIXTURE")
SHA256 = "23e981532ca172c6efab1ea711112ca88d720a88bb40174618dc7c74a7174ace"
LEGACY = ["eva_config", "corda_config", "lora_bias", "exclude_modules",
          "trainable_token_indices"]
NULL_KEYS = ["lora_ga_config", "use_bdlora"]


@unittest.skipUnless(FIXTURE, "SHA-bound external canary adapter not provided")
class SmolNativePeftConstructor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import peft
        from peft import LoraConfig
        from inference.models.transformers import transformers as helper_module

        assert peft.__version__ == os.environ["SMOL_EXPECTED_PEFT_VERSION"] == "0.18.1"
        cls.lora_config = LoraConfig
        cls.helper_module = helper_module
        cls.raw = Path(FIXTURE).read_bytes()
        assert hashlib.sha256(cls.raw).hexdigest() == SHA256
        cls.original = json.loads(cls.raw)
        assert cls.original["peft_version"] == "0.19.1"
        assert all(cls.original[key] is None for key in NULL_KEYS)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="smol-native-peft-")
        self.path = Path(self.directory.name) / "adapter_config.json"
        self.path.write_bytes(self.raw)

    def tearDown(self):
        self.directory.cleanup()

    def test_native_0181_rejects_original_after_legacy_sanitization(self):
        config = {key: value for key, value in self.original.items() if key not in LEGACY}
        with self.assertRaisesRegex(TypeError, "lora_ga_config"):
            self.lora_config(**config)

    def test_native_0181_constructs_exact_adapter_without_semantic_changes(self):
        with patch.object(self.helper_module, "OFFLINE_MODE", True):
            config = self.helper_module.load_compatible_adapter_config(
                str(self.path), unsupported_keys=LEGACY,
                unsupported_null_keys=NULL_KEYS,
            )
        model_config = self.lora_config(**config)
        self.assertEqual(config, {key: value for key, value in self.original.items()
                                  if key not in LEGACY + NULL_KEYS})
        self.assertEqual(model_config.base_model_name_or_path, "smolvlm2")
        self.assertEqual(model_config.revision, "main")
        self.assertEqual(model_config.r, 8)
        self.assertEqual(model_config.lora_alpha, 8)
        self.assertTrue(model_config.use_rslora)
        self.assertTrue(model_config.use_dora)
        self.assertEqual(self.path.read_bytes(), self.raw)

    def test_nondefault_values_fail_closed_before_online_write(self):
        for key in NULL_KEYS:
            for value in (False, True, 0, "", {}, {"enabled": True}):
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(self.original)
                    config[key] = value
                    raw = json.dumps(config).encode()
                    self.path.write_bytes(raw)
                    with patch.object(self.helper_module, "OFFLINE_MODE", False):
                        with self.assertRaisesRegex(ValueError, key):
                            self.helper_module.load_compatible_adapter_config(
                                str(self.path), unsupported_keys=LEGACY,
                                unsupported_null_keys=NULL_KEYS,
                            )
                    self.assertEqual(self.path.read_bytes(), raw)

    def test_unknown_field_still_rejected_by_native_peft(self):
        config = {**self.original, "unknown_future_option": None}
        self.path.write_text(json.dumps(config))
        with patch.object(self.helper_module, "OFFLINE_MODE", True):
            cleaned = self.helper_module.load_compatible_adapter_config(
                str(self.path), unsupported_keys=LEGACY,
                unsupported_null_keys=NULL_KEYS,
            )
        with self.assertRaisesRegex(TypeError, "unknown_future_option"):
            self.lora_config(**cleaned)


if __name__ == "__main__":
    unittest.main()
