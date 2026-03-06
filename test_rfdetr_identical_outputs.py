"""
Test that RF-DETR with transformers 5.x changes produces EXACTLY IDENTICAL outputs
to the original transformers 4.x implementation.

This test contains BOTH implementations (old with head_mask, new without) and
runs them with identical weights to prove outputs are bit-for-bit identical.
"""

import copy
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# SHARED COMPONENTS (identical in both versions)
# =============================================================================

class SharedConfig:
    """Minimal config for testing"""
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        num_windows=1,
        window_block_indexes=None,
        mlp_ratio=4,
        hidden_act="gelu",
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.num_windows = num_windows
        self.window_block_indexes = window_block_indexes if window_block_indexes else list(range(num_hidden_layers))
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_swiglu_ffn = use_swiglu_ffn
        self.out_features = ["stage11"]
        self._attn_implementation = "eager"  # Use eager attention for testing


class LayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state):
        return hidden_state * self.lambda1


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)

    def forward(self, hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# =============================================================================
# OLD IMPLEMENTATION (transformers 4.x style - WITH head_mask)
# =============================================================================

class OldSelfAttention(nn.Module):
    """OLD: Self attention WITH head_mask parameter"""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,  # OLD: has head_mask
        output_attentions: bool = False,
    ):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # OLD: This is the code we removed
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class OldAttention(nn.Module):
    """OLD: Attention module WITH head_mask"""

    def __init__(self, config):
        super().__init__()
        self.attention = OldSelfAttention(config)
        self.output = SelfOutput(config)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,  # OLD: has head_mask
        output_attentions: bool = False,
    ):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class OldLayer(nn.Module):
    """OLD: Transformer layer WITH head_mask"""

    def __init__(self, config):
        super().__init__()
        self.num_windows = config.num_windows
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = OldAttention(config)
        self.layer_scale1 = LayerScale(config)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_scale2 = LayerScale(config)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,  # OLD: has head_mask
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ):
        # OLD: had assertion about head_mask
        assert head_mask is None, "head_mask is not supported for windowed attention"
        assert not output_attentions, "output_attentions is not supported"

        shortcut = hidden_states
        if run_full_attention:
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),
            head_mask,  # OLD: passed head_mask
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            attention_output = attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)

        attention_output = self.layer_scale1(attention_output)
        hidden_states = self.drop_path(attention_output) + shortcut
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)
        layer_output = self.drop_path(layer_output) + hidden_states

        return (layer_output,)


class OldEncoder(nn.Module):
    """OLD: Encoder WITH head_mask"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([OldLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,  # OLD: has head_mask
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > int(self.config.out_features[-1][5:]):
                break

            run_full_attention = i not in self.config.window_block_indexes

            # OLD: extracted head_mask per layer
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                layer_head_mask,  # OLD: passed layer_head_mask
                output_attentions,
                run_full_attention,
            )
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states


# =============================================================================
# NEW IMPLEMENTATION (transformers 5.x style - WITHOUT head_mask)
# =============================================================================

class NewSelfAttention(nn.Module):
    """NEW: Self attention WITHOUT head_mask parameter"""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        # NEW: NO head_mask parameter
        output_attentions: bool = False,
    ):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # NEW: No head_mask code at all

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class NewAttention(nn.Module):
    """NEW: Attention module WITHOUT head_mask"""

    def __init__(self, config):
        super().__init__()
        self.attention = NewSelfAttention(config)
        self.output = SelfOutput(config)

    def forward(
        self,
        hidden_states,
        # NEW: NO head_mask parameter
        output_attentions: bool = False,
    ):
        self_outputs = self.attention(hidden_states, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class NewLayer(nn.Module):
    """NEW: Transformer layer WITHOUT head_mask"""

    def __init__(self, config):
        super().__init__()
        self.num_windows = config.num_windows
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = NewAttention(config)
        self.layer_scale1 = LayerScale(config)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_scale2 = LayerScale(config)

    def forward(
        self,
        hidden_states,
        # NEW: NO head_mask parameter
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ):
        assert not output_attentions, "output_attentions is not supported"

        shortcut = hidden_states
        if run_full_attention:
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),
            # NEW: NO head_mask passed
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            attention_output = attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)

        attention_output = self.layer_scale1(attention_output)
        hidden_states = self.drop_path(attention_output) + shortcut
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)
        layer_output = self.drop_path(layer_output) + hidden_states

        return (layer_output,)


class NewEncoder(nn.Module):
    """NEW: Encoder WITHOUT head_mask"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([NewLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        # NEW: NO head_mask parameter
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > int(self.config.out_features[-1][5:]):
                break

            run_full_attention = i not in self.config.window_block_indexes

            layer_outputs = layer_module(
                hidden_states,
                # NEW: NO head_mask passed
                output_attentions,
                run_full_attention,
            )
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def copy_weights(src_module: nn.Module, dst_module: nn.Module):
    """Copy weights from source to destination module"""
    src_state = src_module.state_dict()
    dst_module.load_state_dict(src_state)


def test_self_attention_identical():
    """Test that OldSelfAttention(head_mask=None) == NewSelfAttention()"""
    print("\n" + "="*70)
    print("TEST 1: SelfAttention - old(head_mask=None) vs new(no head_mask)")
    print("="*70)

    config = SharedConfig()

    # Create both models with same random seed
    torch.manual_seed(42)
    old_model = OldSelfAttention(config)

    torch.manual_seed(42)
    new_model = NewSelfAttention(config)

    old_model.eval()
    new_model.eval()

    # Create test input
    torch.manual_seed(123)
    hidden_states = torch.randn(2, 197, 768)

    # Run both
    with torch.no_grad():
        old_output = old_model(hidden_states, head_mask=None, output_attentions=True)
        new_output = new_model(hidden_states, output_attentions=True)

    # Compare
    context_identical = torch.equal(old_output[0], new_output[0])
    attention_identical = torch.equal(old_output[1], new_output[1])

    print(f"  Context layer identical: {context_identical}")
    print(f"  Attention probs identical: {attention_identical}")

    if context_identical and attention_identical:
        print("  ✓ PASSED: Outputs are BIT-FOR-BIT IDENTICAL")
    else:
        print("  ✗ FAILED: Outputs differ!")
        if not context_identical:
            diff = (old_output[0] - new_output[0]).abs().max().item()
            print(f"    Context max diff: {diff}")
        if not attention_identical:
            diff = (old_output[1] - new_output[1]).abs().max().item()
            print(f"    Attention max diff: {diff}")

    return context_identical and attention_identical


def test_attention_identical():
    """Test that OldAttention(head_mask=None) == NewAttention()"""
    print("\n" + "="*70)
    print("TEST 2: Attention Module - old(head_mask=None) vs new(no head_mask)")
    print("="*70)

    config = SharedConfig()

    torch.manual_seed(42)
    old_model = OldAttention(config)

    torch.manual_seed(42)
    new_model = NewAttention(config)

    old_model.eval()
    new_model.eval()

    torch.manual_seed(123)
    hidden_states = torch.randn(2, 197, 768)

    with torch.no_grad():
        old_output = old_model(hidden_states, head_mask=None, output_attentions=False)
        new_output = new_model(hidden_states, output_attentions=False)

    identical = torch.equal(old_output[0], new_output[0])
    print(f"  Output identical: {identical}")

    if identical:
        print("  ✓ PASSED: Outputs are BIT-FOR-BIT IDENTICAL")
    else:
        diff = (old_output[0] - new_output[0]).abs().max().item()
        print(f"  ✗ FAILED: Max diff = {diff}")

    return identical


def test_layer_identical():
    """Test that OldLayer(head_mask=None) == NewLayer()"""
    print("\n" + "="*70)
    print("TEST 3: Transformer Layer - old(head_mask=None) vs new(no head_mask)")
    print("="*70)

    config = SharedConfig()

    torch.manual_seed(42)
    old_model = OldLayer(config)

    torch.manual_seed(42)
    new_model = NewLayer(config)

    old_model.eval()
    new_model.eval()

    torch.manual_seed(123)
    hidden_states = torch.randn(2, 197, 768)

    with torch.no_grad():
        old_output = old_model(hidden_states, head_mask=None, output_attentions=False, run_full_attention=False)
        new_output = new_model(hidden_states, output_attentions=False, run_full_attention=False)

    identical = torch.equal(old_output[0], new_output[0])
    print(f"  Output identical: {identical}")

    if identical:
        print("  ✓ PASSED: Outputs are BIT-FOR-BIT IDENTICAL")
    else:
        diff = (old_output[0] - new_output[0]).abs().max().item()
        print(f"  ✗ FAILED: Max diff = {diff}")

    return identical


def test_encoder_identical():
    """Test that OldEncoder(head_mask=None) == NewEncoder()"""
    print("\n" + "="*70)
    print("TEST 4: Full Encoder - old(head_mask=None) vs new(no head_mask)")
    print("="*70)

    config = SharedConfig()

    torch.manual_seed(42)
    old_model = OldEncoder(config)

    torch.manual_seed(42)
    new_model = NewEncoder(config)

    old_model.eval()
    new_model.eval()

    torch.manual_seed(123)
    hidden_states = torch.randn(2, 197, 768)

    with torch.no_grad():
        old_output, old_hidden = old_model(hidden_states, head_mask=None, output_hidden_states=True)
        new_output, new_hidden = new_model(hidden_states, output_hidden_states=True)

    output_identical = torch.equal(old_output, new_output)
    print(f"  Final output identical: {output_identical}")

    hidden_identical = True
    for i, (oh, nh) in enumerate(zip(old_hidden, new_hidden)):
        if not torch.equal(oh, nh):
            hidden_identical = False
            diff = (oh - nh).abs().max().item()
            print(f"  Hidden state {i} differs: max diff = {diff}")

    print(f"  All hidden states identical: {hidden_identical}")

    if output_identical and hidden_identical:
        print("  ✓ PASSED: ALL outputs are BIT-FOR-BIT IDENTICAL")
    else:
        print("  ✗ FAILED: Some outputs differ!")

    return output_identical and hidden_identical


def test_sdpa_identical():
    """Test SDPA attention signature change produces identical results"""
    print("\n" + "="*70)
    print("TEST 5: SDPA - old(head_mask as positional) vs new(dropout_p as kwarg)")
    print("="*70)

    torch.manual_seed(42)
    batch_size, num_heads, seq_len, head_dim = 2, 12, 197, 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # OLD: head_mask=None was passed as 4th positional argument
    with torch.no_grad():
        old_output = F.scaled_dot_product_attention(
            query, key, value,
            None,  # head_mask (attn_mask) - was passed as positional
            0.0,   # dropout_p
            is_causal=False,
            scale=None,
        )

    # NEW: dropout_p as keyword argument, no attn_mask
    with torch.no_grad():
        new_output = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )

    identical = torch.equal(old_output, new_output)
    print(f"  SDPA output identical: {identical}")

    if identical:
        print("  ✓ PASSED: Outputs are BIT-FOR-BIT IDENTICAL")
    else:
        diff = (old_output - new_output).abs().max().item()
        print(f"  ✗ FAILED: Max diff = {diff}")

    return identical


def main():
    print("="*70)
    print("RF-DETR TRANSFORMERS 5.x COMPATIBILITY TEST")
    print("="*70)
    print("\nThis test proves that removing head_mask produces IDENTICAL outputs")
    print("because head_mask was ALWAYS None in RF-DETR usage.\n")

    results = []

    results.append(("SelfAttention", test_self_attention_identical()))
    results.append(("Attention", test_attention_identical()))
    results.append(("Layer", test_layer_identical()))
    results.append(("Encoder", test_encoder_identical()))
    results.append(("SDPA", test_sdpa_identical()))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("="*70)
        print("ALL TESTS PASSED!")
        print("The transformers 5.x changes produce EXACTLY IDENTICAL outputs.")
        print("="*70)
        return 0
    else:
        print("="*70)
        print("SOME TESTS FAILED!")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
