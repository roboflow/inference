import unittest
import torch
from lidra.model.backbone.dit.embedder.dino import Dino
from lidra.model.backbone.dit.embedder.moge import MoGe
from lidra.model.backbone.dit.embedder.siglip2 import SigLIP2
from lidra.model.backbone.dit.embedder.embedder_fuser import EmbedderFuser
from lidra.test.util import run_unittest, run_only_if_cuda_is_available


class UnitTests(unittest.TestCase):
    IMAGE_CHANNELS = 3
    INPUT_N_TOKENS = 32
    CONDITION_N_TOKENS = 16

    def _check_output(self, input_tensor, output_tensor):
        self.assertEqual(output_tensor.shape, input_tensor.shape)

    def _create_condition(self, condition_embedder):
        return torch.rand(
            (
                3,
                condition_embedder.input_channels,
                condition_embedder.input_size,
                condition_embedder.input_size,
            )
        )

    def test_moge_init(self):
        _ = MoGe()

    @run_only_if_cuda_is_available(default_device="cuda")
    def test_moge(self):
        moge_cond_embedder = MoGe().cuda()
        input_image = self._create_condition(moge_cond_embedder)
        output_tensor = moge_cond_embedder(input_image)
        self._check_output(output_tensor, torch.randn((3, 1370, 1024)))

    @run_only_if_cuda_is_available(default_device="cuda")
    def test_fuser(self):
        dino_embedder = Dino(input_size=518).cuda()
        moge_cond_embedder = MoGe().cuda()
        input_dict = {
            "image1": self._create_condition(moge_cond_embedder),
            "image2": self._create_condition(moge_cond_embedder),
            "mask1": self._create_condition(moge_cond_embedder),
        }
        embedder_list = [
            (dino_embedder, [("image1", 0), ("image2", 1), ("mask1", 2)]),
            (moge_cond_embedder, [("image1", 0), ("image2", 1)]),
        ]
        embedder_fuser = EmbedderFuser(
            embedder_list=embedder_list, use_pos_embedding="random"
        )
        output_tensor = embedder_fuser(**input_dict)
        self._check_output(output_tensor, torch.randn((3, 1370 * 5, 1024)))
        embedder_fuser = EmbedderFuser(
            embedder_list=embedder_list, use_pos_embedding="learned"
        )
        output_tensor = embedder_fuser(**input_dict)
        self._check_output(output_tensor, torch.randn((3, 1370 * 5, 1024)))

    def test_siglip2_init(self):
        _ = SigLIP2()

    @run_only_if_cuda_is_available(default_device="cuda")
    def test_siglip2(self):
        siglip_cond_embedder = SigLIP2().cuda()
        input_image = self._create_condition(siglip_cond_embedder)
        output_tensor = siglip_cond_embedder(input_image)
        self._check_output(output_tensor, torch.randn((3, 1, 1024)))


class DropoutTests(unittest.TestCase):
    """Test modality dropout functionality in EmbedderFuser."""

    def setUp(self):
        """Create mock embedders and test data."""
        # Create simple mock embedders
        self.mock_embedder = self._create_mock_embedder(embed_dim=256)
        self.batch_size = 4
        self.seq_len = 10
        self.embed_dim = 256

    def _create_mock_embedder(self, embed_dim):
        """Create a mock embedder that returns fixed-size tensors."""

        class MockEmbedder(torch.nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.embed_dim = embed_dim
                self.input_channels = 3
                self.input_size = 224

            def forward(self, x):
                batch_size = x.shape[0]
                return torch.ones(batch_size, 10, embed_dim)  # Fixed seq_len=10

        return MockEmbedder(embed_dim)

    def _create_fuser_with_dropout(self, dropout_prob, drop_modalities_weight):
        """Helper to create EmbedderFuser with dropout configuration."""
        embedder_list = [
            (self.mock_embedder, [("image", "group1")]),
            (self.mock_embedder, [("mask", "group2")]),
            (self.mock_embedder, [("pointmap", "group3")]),
        ]
        return EmbedderFuser(
            embedder_list=embedder_list,
            use_pos_embedding="learned",
            dropout_prob=dropout_prob,
            drop_modalities_weight=drop_modalities_weight,
            projection_net_hidden_dim_multiplier=0,  # Skip projection for simplicity
        )

    def test_no_dropout_when_eval_mode(self):
        """Verify no dropout is applied in eval mode."""
        fuser = self._create_fuser_with_dropout(
            dropout_prob=0.5, drop_modalities_weight=[(["image", "mask"], 1.0)]
        )
        fuser.eval()

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        output = fuser(**inputs)
        # All tokens should be non-zero in eval mode
        self.assertTrue((output != 0).all())

    def test_no_dropout_when_prob_zero(self):
        """Verify no dropout when dropout_prob=0."""
        fuser = self._create_fuser_with_dropout(
            dropout_prob=0.0, drop_modalities_weight=[(["image"], 1.0)]
        )
        fuser.train()

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        output = fuser(**inputs)
        # All tokens should be non-zero when dropout_prob=0
        self.assertTrue((output != 0).all())

    def test_dropout_applies_in_training(self):
        """Verify dropout is applied during training with multiple runs."""
        fuser = self._create_fuser_with_dropout(
            dropout_prob=0.5, drop_modalities_weight=[(["pointmap"], 1.0)]
        )
        fuser.train()

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        # Run multiple times to check stochasticity
        dropout_occurred = False
        for _ in range(10):
            output = fuser(**inputs)
            # Check if pointmap tokens (last seq_len tokens) are dropped for any batch element
            pointmap_tokens = output[:, -self.seq_len :, :]
            if (pointmap_tokens == 0).any():
                dropout_occurred = True
                break

        self.assertTrue(
            dropout_occurred, "Dropout should occur at least once in 10 runs"
        )

    def test_independent_dropout_per_batch(self):
        """Verify dropout is independent across batch samples."""
        fuser = self._create_fuser_with_dropout(
            dropout_prob=0.5, drop_modalities_weight=[(["image"], 1.0)]
        )
        fuser.train()

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        # Run multiple times to collect dropout patterns
        patterns = []
        for _ in range(20):
            output = fuser(**inputs)
            # Check which batch elements have image dropped (first seq_len tokens)
            image_tokens = output[:, : self.seq_len, :]
            batch_pattern = (image_tokens.sum(dim=[1, 2]) == 0).cpu()
            patterns.append(batch_pattern)

        # Check that not all batch elements have the same pattern every time
        patterns = torch.stack(patterns)
        # At least one batch element should have different patterns
        variance_per_batch = patterns.float().var(dim=0)
        self.assertTrue(
            (variance_per_batch > 0).any(),
            "Dropout should be independent per batch element",
        )

    def test_dropout_preserves_shape(self):
        """Verify output shape is preserved with dropout."""
        fuser = self._create_fuser_with_dropout(
            dropout_prob=0.8,
            drop_modalities_weight=[
                (["image", "mask"], 0.5),
                (["pointmap"], 0.5),
            ],
        )
        fuser.train()

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        output = fuser(**inputs)
        expected_shape = (self.batch_size, 3 * self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_multiple_dropout_configurations(self):
        """Test that different dropout configurations are applied correctly."""
        fuser = self._create_fuser_with_dropout(
            dropout_prob=0.9,  # High probability to ensure we see both configs
            drop_modalities_weight=[
                (["image"], 0.5),
                (["mask", "pointmap"], 0.5),
            ],
        )
        fuser.train()

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        # Run multiple times and check we see both configurations
        config1_seen = False  # Only image dropped
        config2_seen = False  # Mask and pointmap dropped

        for _ in range(30):
            output = fuser(**inputs)

            for batch_idx in range(self.batch_size):
                batch_output = output[batch_idx]
                image_dropped = (batch_output[: self.seq_len] == 0).all()
                mask_dropped = (
                    batch_output[self.seq_len : 2 * self.seq_len] == 0
                ).all()
                pointmap_dropped = (batch_output[2 * self.seq_len :] == 0).all()

                if image_dropped and not mask_dropped and not pointmap_dropped:
                    config1_seen = True
                elif not image_dropped and mask_dropped and pointmap_dropped:
                    config2_seen = True

                if config1_seen and config2_seen:
                    break

            if config1_seen and config2_seen:
                break

        self.assertTrue(config1_seen, "Should see config with only image dropped")
        self.assertTrue(
            config2_seen, "Should see config with mask and pointmap dropped"
        )

    def test_dropout_weight_distribution(self):
        """Verify dropout follows configured weight distribution."""
        # Use high dropout prob and many samples to test distribution
        fuser = self._create_fuser_with_dropout(
            dropout_prob=0.8,
            drop_modalities_weight=[
                (["image"], 0.75),  # Should occur ~60% of dropout cases (0.8 * 0.75)
                (["mask"], 0.25),  # Should occur ~20% of dropout cases (0.8 * 0.25)
            ],
        )
        fuser.train()

        inputs = {
            "image": torch.ones(100, 3, 224, 224),  # Large batch for statistics
            "mask": torch.ones(100, 3, 224, 224),
            "pointmap": torch.ones(100, 3, 224, 224),
        }

        # Count occurrences over multiple runs
        total_samples = 0
        image_dropped_count = 0
        mask_dropped_count = 0
        no_dropout_count = 0

        for _ in range(10):
            output = fuser(**inputs)

            for batch_idx in range(100):
                total_samples += 1
                batch_output = output[batch_idx]
                image_dropped = (batch_output[: self.seq_len] == 0).all()
                mask_dropped = (
                    batch_output[self.seq_len : 2 * self.seq_len] == 0
                ).all()

                if image_dropped:
                    image_dropped_count += 1
                elif mask_dropped:
                    mask_dropped_count += 1
                else:
                    no_dropout_count += 1

        # Check proportions (with some tolerance for randomness)
        image_ratio = image_dropped_count / total_samples
        mask_ratio = mask_dropped_count / total_samples
        no_dropout_ratio = no_dropout_count / total_samples

        self.assertAlmostEqual(image_ratio, 0.6, delta=0.05)
        self.assertAlmostEqual(mask_ratio, 0.2, delta=0.05)
        self.assertAlmostEqual(no_dropout_ratio, 0.2, delta=0.05)

    def test_force_drop_always_applied(self):
        """Verify force_drop_modalities works in both train and eval modes."""
        embedder_list = [
            (self.mock_embedder, [("image", "group1")]),
            (self.mock_embedder, [("mask", "group2")]),
            (self.mock_embedder, [("pointmap", "group3")]),
        ]
        fuser = EmbedderFuser(
            embedder_list=embedder_list,
            use_pos_embedding="learned",
            dropout_prob=0.0,  # No probabilistic dropout
            drop_modalities_weight=None,
            force_drop_modalities=["mask", "pointmap"],
            projection_net_hidden_dim_multiplier=0,
        )

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        # Test in eval mode
        fuser.eval()
        output = fuser(**inputs)
        image_tokens = output[:, : self.seq_len, :]
        mask_tokens = output[:, self.seq_len : 2 * self.seq_len, :]
        pointmap_tokens = output[:, 2 * self.seq_len :, :]

        self.assertTrue((image_tokens != 0).any(), "Image should not be dropped")
        self.assertTrue((mask_tokens == 0).all(), "Mask should be force dropped")
        self.assertTrue(
            (pointmap_tokens == 0).all(), "Pointmap should be force dropped"
        )

        # Test in train mode
        fuser.train()
        output = fuser(**inputs)
        image_tokens = output[:, : self.seq_len, :]
        mask_tokens = output[:, self.seq_len : 2 * self.seq_len, :]
        pointmap_tokens = output[:, 2 * self.seq_len :, :]

        self.assertTrue((image_tokens != 0).any(), "Image should not be dropped")
        self.assertTrue((mask_tokens == 0).all(), "Mask should be force dropped")
        self.assertTrue(
            (pointmap_tokens == 0).all(), "Pointmap should be force dropped"
        )

    def test_force_drop_with_probabilistic_dropout(self):
        """Verify force_drop and probabilistic dropout work together."""
        embedder_list = [
            (self.mock_embedder, [("image", "group1")]),
            (self.mock_embedder, [("mask", "group2")]),
            (self.mock_embedder, [("pointmap", "group3")]),
        ]
        fuser = EmbedderFuser(
            embedder_list=embedder_list,
            use_pos_embedding="learned",
            dropout_prob=0.9,  # High probability
            drop_modalities_weight=[
                (["image"], 1.0)  # Only drop image probabilistically
            ],
            force_drop_modalities=["mask"],  # Always drop mask
            projection_net_hidden_dim_multiplier=0,
        )
        fuser.train()

        inputs = {
            "image": torch.ones(self.batch_size, 3, 224, 224),
            "mask": torch.ones(self.batch_size, 3, 224, 224),
            "pointmap": torch.ones(self.batch_size, 3, 224, 224),
        }

        # Run multiple times to check behavior
        for _ in range(10):
            output = fuser(**inputs)
            mask_tokens = output[:, self.seq_len : 2 * self.seq_len, :]
            pointmap_tokens = output[:, 2 * self.seq_len :, :]

            # Mask should ALWAYS be dropped (force_drop)
            self.assertTrue(
                (mask_tokens == 0).all(), "Mask should always be force dropped"
            )
            # Pointmap should NEVER be dropped (not in either list)
            self.assertTrue(
                (pointmap_tokens != 0).all(), "Pointmap should never be dropped"
            )
            # Image might be dropped (probabilistic)


if __name__ == "__main__":
    run_unittest(UnitTests)
    run_unittest(DropoutTests)
