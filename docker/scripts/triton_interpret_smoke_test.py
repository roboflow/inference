import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> None:
    x = torch.arange(0, 32, dtype=torch.float32)
    y = torch.full((32,), 2.0, dtype=torch.float32)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["block_size"]),)
    add_kernel[grid](x, y, out, x.numel(), block_size=16)
    expected = x + y
    if not torch.allclose(out, expected):
        raise SystemExit(f"unexpected output: {out.tolist()} != {expected.tolist()}")
    print(
        "triton smoke test passed:",
        {
            "triton": triton.__version__,
            "torch": torch.__version__,
            "result_tail": out[-4:].tolist(),
        },
    )


if __name__ == "__main__":
    main()
