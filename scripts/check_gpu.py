"""Print the Python/torch/CUDA versions and exercise bf16 matmul + SDPA on the GPU."""
import sys

import torch


def main():
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        if torch.version.cuda is not None:
            sys.exit("ERROR: this is a CUDA build of torch but no GPU is visible (driver? CUDA_VISIBLE_DEVICES?). "
                     "Training would refuse to start.")
        print("CPU build of torch: fine for smoke tests only (bash scripts/smoke_test.sh --device cpu).")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}"
          f"  archs {torch.cuda.get_arch_list()}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM free {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(2, 12, 197, 64, device="cuda", dtype=torch.bfloat16)
    y = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    ok = bool(torch.isfinite((x @ x).float()).all()) and bool(torch.isfinite(y.float()).all())
    print(f"bf16 matmul + SDPA: {'OK' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
