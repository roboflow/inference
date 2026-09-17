# Meta V-JEPA 2.1

Source: https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812

The adjacent MIT LICENSE applies to these files.
Original copyright notices remain in each file.

- `encoder.py`: `app/vjepa_2_1/models/vision_transformer.py`, retaining VisionTransformer and vit_base.
- `modules.py`: `app/vjepa_2_1/models/utils/modules.py`, excluding Lambda_LinearWarmupHold.
- `patch_embed.py`: `app/vjepa_2_1/models/utils/patch_embed.py`, excluding AudioPatchEmbed and its einops import.
- `tensors.py`: `src/utils/tensors.py`, excluding repeat_interleave_batch.
- `masks.py`: `src/masks/utils.py`.

Local changes: package-relative imports, formatting, and one FP16 compatibility adjustment in RoPEAttention.
That adjustment casts rotated queries and keys to the values' FP16 dtype before SDPA, which requires matching input dtypes.
It leaves the established BF16 path unchanged.
The encoder uses the V-JEPA 2.1 implementation, not the older V-JEPA 2 class.
CrossAttentionBlock has the same computation as the Meta block used by the benchmark head.
Builds and training do not fetch upstream source.

This serving copy matches roboflow-train commit e8486ae7, with formatting and import ordering only.
The adjacent serving head keeps the encoder/head state names from that trainer.
