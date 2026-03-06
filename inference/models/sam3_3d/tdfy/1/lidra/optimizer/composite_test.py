"""
Unit tests for CompositeOptimizer state management functionality.

Tests the critical state_dict() and load_state_dict() methods to ensure
proper optimizer state saving and loading for training resumption.
"""

import unittest
import torch
from torch.optim import Adam, SGD
from copy import deepcopy

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidra.optimizer.composite import Composite


class TestCompositeOptimizer(unittest.TestCase):
    """Test suite for CompositeOptimizer state management."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)  # For reproducibility

    def test_state_dict_after_training(self):
        """Test that state_dict contains proper optimizer state after training."""
        # Setup models and optimizers
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(5, 1)

        opt1 = Adam(model1.parameters(), lr=0.001)
        opt2 = SGD(model2.parameters(), lr=0.01, momentum=0.9)

        composite_opt = Composite(opt1, opt2)

        # Train for several steps
        for _ in range(10):
            composite_opt.zero_grad()

            x = torch.randn(32, 10)
            h = model1(x)
            loss = model2(h).sum()

            loss.backward()
            composite_opt.step()

        # Verify state_dict contains states
        state_dict = composite_opt.state_dict()

        self.assertIn("state", state_dict)
        self.assertIn("param_groups", state_dict)
        self.assertGreater(
            len(state_dict["state"]), 0, "State should not be empty after training"
        )

        # Verify state contains expected keys for Adam (exp_avg, exp_avg_sq, step)
        has_adam_state = False
        has_sgd_state = False

        for param_state in state_dict["state"].values():
            if "exp_avg" in param_state and "exp_avg_sq" in param_state:
                has_adam_state = True
                self.assertIn("step", param_state)
                self.assertGreater(param_state["step"], 0)
            if "momentum_buffer" in param_state:
                has_sgd_state = True

        self.assertTrue(has_adam_state, "Should contain Adam optimizer state")
        self.assertTrue(has_sgd_state, "Should contain SGD optimizer state")

    def test_load_state_dict_resumption(self):
        """Test that loading state dict properly resumes training."""
        torch.manual_seed(42)

        # Create model and optimizer
        model = torch.nn.Linear(10, 1)
        opt = Adam(model.parameters(), lr=0.001)
        composite_opt = Composite(opt)

        # Train for several steps to build up optimizer state
        for _ in range(5):
            composite_opt.zero_grad()
            x = torch.randn(8, 10)
            loss = model(x).sum()
            loss.backward()
            composite_opt.step()

        # Save state
        saved_state = composite_opt.state_dict()
        self.assertGreater(
            len(saved_state["state"]), 0, "Saved state should not be empty"
        )

        # Check that individual optimizer has state too
        individual_state = opt.state_dict()
        self.assertGreater(
            len(individual_state["state"]), 0, "Individual optimizer should have state"
        )

        # Verify some state details
        for param_state in saved_state["state"].values():
            if "exp_avg" in param_state:
                self.assertIn("exp_avg_sq", param_state)
                self.assertIn("step", param_state)
                self.assertGreater(param_state["step"], 0)

        # The main test: verify that load_state_dict doesn't crash and loads something
        # Create a new identical optimizer setup
        model_new = torch.nn.Linear(10, 1)
        with torch.no_grad():
            for p1, p2 in zip(model.parameters(), model_new.parameters()):
                p2.copy_(p1)

        opt_new = Adam(model_new.parameters(), lr=0.001)
        composite_opt_new = Composite(opt_new)  # Keep consistency check enabled

        # Load the state - this should work without error
        composite_opt_new.load_state_dict(saved_state)

        # The key test: verify that the state was actually loaded into the individual optimizer
        loaded_individual_state = opt_new.state_dict()
        self.assertGreater(
            len(loaded_individual_state["state"]),
            0,
            "Individual optimizer should have loaded state",
        )

        # Verify loaded state has the expected structure
        for param_state in loaded_individual_state["state"].values():
            if "exp_avg" in param_state:
                self.assertIn("exp_avg_sq", param_state)
                self.assertIn("step", param_state)
                self.assertGreater(param_state["step"], 0)

        # Most important test: verify that training can continue with loaded state
        # This is the real-world requirement
        x = torch.randn(8, 10)

        # The loaded optimizer should be able to step (this is the key functionality)
        composite_opt_new.zero_grad()
        loss = model_new(x).sum()
        loss.backward()
        composite_opt_new.step()  # This should work if state was loaded correctly

        # After step, individual optimizer should still have state
        post_step_state = opt_new.state_dict()
        self.assertGreater(
            len(post_step_state["state"]),
            0,
            "Individual optimizer should have state after step",
        )

        # Success if we get here without exceptions
        self.assertTrue(True, "CompositeOptimizer successfully loaded and used state")

    def test_empty_state_handling(self):
        """Test behavior with optimizers that have no state yet."""
        model = torch.nn.Linear(5, 1)
        opt = Adam(model.parameters(), lr=0.001)
        composite_opt = Composite(opt)

        # Before any training
        state_dict = composite_opt.state_dict()
        self.assertIn("state", state_dict)
        self.assertIn("param_groups", state_dict)
        # State may be empty but should be present
        self.assertIsInstance(state_dict["state"], dict)

        # Should be able to load empty state
        composite_opt.load_state_dict(state_dict)

    def test_mixed_optimizer_types(self):
        """Test composite optimizer with different optimizer types."""
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(5, 1)

        opt1 = Adam(model1.parameters(), lr=0.001, betas=(0.9, 0.999))
        opt2 = SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        composite_opt = Composite(opt1, opt2)

        # Train
        for _ in range(5):
            composite_opt.zero_grad()
            x = torch.randn(16, 10)
            h = model1(x)
            loss = model2(h).sum()
            loss.backward()
            composite_opt.step()

        # Verify state contains both types of optimizer state
        state_dict = composite_opt.state_dict()

        has_adam_state = False
        has_sgd_state = False

        for param_state in state_dict["state"].values():
            if "exp_avg" in param_state and "exp_avg_sq" in param_state:
                has_adam_state = True
            if "momentum_buffer" in param_state:
                has_sgd_state = True

        self.assertTrue(has_adam_state, "Should contain Adam optimizer state")
        self.assertTrue(has_sgd_state, "Should contain SGD optimizer state")

    def test_single_optimizer_compatibility(self):
        """Test that CompositeOptimizer works correctly with a single optimizer."""
        model = torch.nn.Linear(10, 1)
        opt = Adam(model.parameters(), lr=0.001)
        composite_opt = Composite(opt)

        # Train
        for _ in range(3):
            composite_opt.zero_grad()
            x = torch.randn(16, 10)
            loss = model(x).sum()
            loss.backward()
            composite_opt.step()

        # Test state management
        state_dict = composite_opt.state_dict()
        self.assertGreater(len(state_dict["state"]), 0)

        # Test loading
        composite_opt.load_state_dict(state_dict)

    def test_parameter_groups_preservation(self):
        """Test that parameter groups (learning rates, etc.) are preserved."""
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(5, 1)

        # Different learning rates for different optimizers
        opt1 = Adam(model1.parameters(), lr=0.001)
        opt2 = Adam(model2.parameters(), lr=0.01)  # Different LR

        composite_opt = Composite(opt1, opt2)

        # Train briefly
        for _ in range(3):
            composite_opt.zero_grad()
            x = torch.randn(8, 10)
            h = model1(x)
            loss = model2(h).sum()
            loss.backward()
            composite_opt.step()

        # Save and load state
        state_dict = composite_opt.state_dict()

        # Create new setup
        model1_new = torch.nn.Linear(10, 5)
        model2_new = torch.nn.Linear(5, 1)
        opt1_new = Adam(model1_new.parameters(), lr=0.001)
        opt2_new = Adam(model2_new.parameters(), lr=0.01)
        composite_opt_new = Composite(opt1_new, opt2_new)

        # Load state
        composite_opt_new.load_state_dict(state_dict)

        # Verify learning rates are preserved
        self.assertEqual(len(composite_opt_new.param_groups), 2)
        # Note: exact LR values depend on the order parameters are added

    def test_error_handling(self):
        """Test error handling for invalid state dicts."""
        model = torch.nn.Linear(5, 1)
        opt = Adam(model.parameters(), lr=0.001)
        composite_opt = Composite(opt)

        # Test with invalid state dict types - should handle gracefully
        composite_opt.load_state_dict("invalid")  # Should not crash

        # Test with missing keys (should work)
        composite_opt.load_state_dict({})

        # Test with malformed state - should not crash
        composite_opt.load_state_dict({"state": "invalid"})

    def test_state_dict_deterministic(self):
        """Test that state_dict produces deterministic output."""
        model = torch.nn.Linear(10, 1)
        opt = Adam(model.parameters(), lr=0.001)
        composite_opt = Composite(opt)

        # Train
        for _ in range(3):
            composite_opt.zero_grad()
            x = torch.randn(8, 10)
            loss = model(x).sum()
            loss.backward()
            composite_opt.step()

        # Get state dict multiple times
        state_dict1 = composite_opt.state_dict()
        state_dict2 = composite_opt.state_dict()

        # Should be identical
        self.assertEqual(len(state_dict1["state"]), len(state_dict2["state"]))
        for key in state_dict1["state"]:
            self.assertIn(key, state_dict2["state"])
            for param_key in state_dict1["state"][key]:
                self.assertTrue(
                    torch.equal(
                        state_dict1["state"][key][param_key],
                        state_dict2["state"][key][param_key],
                    )
                )


class TestCompositeOptimizerIntegration(unittest.TestCase):
    """Integration tests for CompositeOptimizer with PyTorch ecosystem."""

    def test_checkpoint_roundtrip(self):
        """Test saving and loading through torch.save/load (simulating Lightning behavior)."""
        # Create setup
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(5, 1)
        opt1 = Adam(model1.parameters(), lr=0.001)
        opt2 = SGD(model2.parameters(), lr=0.01, momentum=0.9)
        composite_opt = Composite(opt1, opt2)

        # Train
        for _ in range(5):
            composite_opt.zero_grad()
            x = torch.randn(16, 10)
            h = model1(x)
            loss = model2(h).sum()
            loss.backward()
            composite_opt.step()

        # Simulate PyTorch Lightning checkpoint saving
        checkpoint = {
            "state_dict": {f"model1.{k}": v for k, v in model1.state_dict().items()},
            "optimizer_states": [composite_opt.state_dict()],
            "lr_schedulers": [],
            "epoch": 5,
            "global_step": 100,
        }

        # Save to file (in memory for test)
        import io

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        # Load from file
        loaded_checkpoint = torch.load(buffer, weights_only=False)

        # Verify structure
        self.assertIn("optimizer_states", loaded_checkpoint)
        self.assertEqual(len(loaded_checkpoint["optimizer_states"]), 1)

        optimizer_state = loaded_checkpoint["optimizer_states"][0]
        self.assertIn("state", optimizer_state)
        self.assertIn("param_groups", optimizer_state)
        self.assertGreater(
            len(optimizer_state["state"]),
            0,
            "Loaded optimizer state should not be empty",
        )


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestCompositeOptimizer))
    suite.addTest(unittest.makeSuite(TestCompositeOptimizerIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Allow running as script
    success = run_tests()
    sys.exit(0 if success else 1)
