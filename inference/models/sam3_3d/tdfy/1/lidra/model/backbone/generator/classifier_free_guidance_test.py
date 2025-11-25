import unittest
import torch
import numpy as np

from lidra.model.backbone.generator.classifier_free_guidance import (
    ClassifierFreeGuidance,
)
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    def test_classifier_free_guidance(self):
        def mock_backbone(x, t, c=None, **kwargs):
            cfg_activate = kwargs.pop("cfg", False)
            if cfg_activate:
                # special customized handling
                return x - 1
            if c is None:
                return x
            if type(c) == np.array:
                c = torch.tensor(c)
            return x + c

        x = torch.tensor(1.0)
        t = None
        c = torch.tensor(2.0)

        # test always unconditional (zeros)
        cfg = ClassifierFreeGuidance(
            backbone=mock_backbone,
            p_unconditional=1.0,
            strength=3,
            unconditional_handling="zeros",
        )
        cfg.train(True)
        self.assertEqual(cfg(x, t, c).item(), 1.0)  # c should be set to 0, always
        self.assertRaises(RuntimeError, lambda: cfg(x, t))
        cfg.train(False)
        self.assertEqual(cfg(x, t, c).item(), 9.0)  # (1 + 3) * 3 - 3 * 1 = 9
        self.assertRaises(RuntimeError, lambda: cfg(x, t))

        # test always conditional (zeros)
        cfg = ClassifierFreeGuidance(
            backbone=mock_backbone,
            p_unconditional=0.0,
            strength=3,
            unconditional_handling="zeros",
        )
        cfg.train(True)
        self.assertEqual(cfg(x, t, c).item(), 3.0)  # c should never be set to 0, always
        self.assertRaises(RuntimeError, lambda: cfg(x, t))
        cfg.train(False)
        self.assertEqual(cfg(x, t, c).item(), 9.0)  # (1 + 3) * 3 - 3 * 1 = 9
        self.assertRaises(RuntimeError, lambda: cfg(x, t))

        # test always unconditional (discard)
        cfg = ClassifierFreeGuidance(
            backbone=mock_backbone,
            p_unconditional=1.0,
            strength=3,
            unconditional_handling="discard",
        )
        cfg.train(True)
        self.assertEqual(cfg(x, t, c).item(), 1.0)  # c should be set to 0, always
        self.assertEqual(cfg(x, t).item(), 1.0)
        cfg.train(False)
        self.assertEqual(cfg(x, t, c).item(), 9.0)  # (1 + 3) * 3 - 3 * 1 = 9
        self.assertEqual(cfg(x, t).item(), 1.0)

        # test always conditional (discard)
        cfg = ClassifierFreeGuidance(
            backbone=mock_backbone,
            p_unconditional=0.0,
            strength=3,
            unconditional_handling="discard",
        )
        cfg.train(True)
        self.assertEqual(cfg(x, t, c).item(), 3.0)  # c should never be set to 0, always
        self.assertEqual(cfg(x, t).item(), 1.0)
        cfg.train(False)
        self.assertEqual(cfg(x, t, c).item(), 9.0)  # (1 + 3) * 3 - 3 * 1 = 9
        self.assertEqual(cfg(x, t).item(), 1.0)

        # test always unconditional (discard)
        cfg = ClassifierFreeGuidance(
            backbone=mock_backbone,
            p_unconditional=1.0,
            strength=3,
            unconditional_handling="drop_tensors",
        )
        cfg.train(True)
        self.assertEqual(cfg(x, t, c).item(), 1.0)  # c should be set to 0, always
        self.assertEqual(cfg(x, t, c.numpy()).item(), 3.0)  # c should not be dropped
        self.assertRaises(RuntimeError, lambda: cfg(x, t))
        cfg.train(False)
        self.assertEqual(cfg(x, t, c).item(), 9.0)  # (1 + 3) * 3 - 3 * 1 = 9
        self.assertRaises(RuntimeError, lambda: cfg(x, t))

        # test always unconditional (discard)
        cfg = ClassifierFreeGuidance(
            backbone=mock_backbone,
            p_unconditional=1.0,
            strength=3,
            unconditional_handling="add_flag",
        )
        cfg.train(True)
        self.assertEqual(cfg(x, t, c).item(), 0.0)  # minus 1 for this
        self.assertRaises(RuntimeError, lambda: cfg(x, t))
        cfg.train(False)
        self.assertEqual(cfg(x, t, c).item(), 12.0)  # (1 + 3) * 3 - 3 * 0 = 12
        self.assertRaises(RuntimeError, lambda: cfg(x, t))


if __name__ == "__main__":
    run_unittest(UnitTests)
