# Meta V-JEPA 2.1

Source: https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812

`architecture.py` adapts the video inference path from these Meta files:

- `app/vjepa_2_1/models/vision_transformer.py`
- `app/vjepa_2_1/models/utils/modules.py`
- `app/vjepa_2_1/models/utils/patch_embed.py`

The adjacent MIT LICENSE applies to the adapted code.

Only ViT-B video inference and the head's cross-attention block remain.
The serving wrapper validates the exported architecture arguments before construction.
The encoder retains the original parameter names, including unused image embeddings and intermediate normalization layers, for strict loading of existing trainer artifacts.
Meta's spatial RoPE interpolation, channel rotation layout, normalization constants, and attention projections remain unchanged.
Rotated queries and keys retain the prior FP16 cast before SDPA.
The BF16 path remains unchanged.

Training initialization, token masking, alternate architectures, image inference, activation checkpointing, and intermediate feature outputs are omitted.
All learned parameters come from the required checkpoint.
The full reference implementation remains in git history at inference commit `7c2651265a4248a58eb7cf2c277c328b704d2595`.
Builds and serving do not fetch upstream source.
