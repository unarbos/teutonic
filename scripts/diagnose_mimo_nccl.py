"""Model-free NCCL reduction check for the MiMo determinism investigation."""
import json
import os
from pathlib import Path

os.environ.update(NCCL_COLLNET_ENABLE="0", NCCL_NVLS_ENABLE="0", NCCL_P2P_NET_DISABLE="1",
                  NCCL_MIN_NCHANNELS="1", NCCL_MAX_NCHANNELS="1", NCCL_PROTO="Simple",
                  NCCL_ALGO="ring,tree;allreduce:tree", NCCL_NTHREADS="1",
                  NCCL_SOCKET_NTHREADS="1", NCCL_LAUNCH_MODE="GROUP")

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(100 + rank)
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        source = torch.randn((8192, 3072), device="cuda", dtype=dtype)
        reference = None
        for size in (8192, 128, 512, 721, 1024, 2048, 4096, 8192):
            value = source[:size].clone()
            dist.all_reduce(value)
            sample = value[:128].float().cpu()
            if reference is None:
                reference = sample.clone()
            delta = (sample - reference).abs()
            results.append({"dtype": str(dtype), "rows": size,
                            "different": int((delta != 0).sum()),
                            "max_abs": float(delta.max())})
    root = Path(os.environ["TEUTONIC_NCCL_DIAGNOSTIC_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / f"rank-{rank}.json").write_text(json.dumps(results, indent=2))
    print(rank, results, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
