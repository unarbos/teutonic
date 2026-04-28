#!/usr/bin/env python3
"""Side-load smoke for the new trainability_probe.

Run on the GPU box. Loads /root/eval_torch.NEW.py via importlib (so the live
/root/eval_torch.py used by the running eval_server is untouched), pulls the
current live king onto a single GPU, and verifies:

  1. honest king passes all 5 layers (ok=True, status="ok")
  2. after pumping RMSNorm weights by 1000x the same king fails Layer 1
     (norm_weight_cap), no compute spent
  3. after pumping a single projection group by a large factor the same king
     fails Layer 4 (param_group_grad)
  4. resets and re-probes -> ok=True again (snapshot/restore is intact)

Usage (on GPU box):
    cd /root && . .venv/bin/activate && . env.sh
    python /root/_probe_smoke.py --king <repo> --gpu 7
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch


def load_module(path):
    spec = importlib.util.spec_from_file_location("eval_torch_new", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["eval_torch_new"] = m
    spec.loader.exec_module(m)
    return m


def summarize(verdict):
    keep = (
        "ok", "status", "reason",
        "max_norm_weight", "global_grad_norm",
        "param_group_grad_norms", "norm_quantization",
        "n_seeds", "loss_before",
    )
    return {k: verdict.get(k) for k in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="/root/eval_torch.NEW.py")
    ap.add_argument("--king", required=True, help="Live king HF repo")
    ap.add_argument("--king-revision", default=None)
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"[smoke] loading new eval_torch from {args.module}")
    m = load_module(args.module)

    print(f"[smoke] loading king {args.king} on {device}")
    t0 = time.time()
    model = m.load_model(args.king, device, label="probe-king",
                          force_download=False, revision=args.king_revision)
    print(f"[smoke] king loaded in {time.time() - t0:.1f}s")

    print("\n[smoke] === TEST 1: honest king ===")
    v1 = m.trainability_probe(model)
    print(json.dumps(summarize(v1), default=str, indent=2))
    assert v1["ok"] is True, f"honest king should pass: {v1.get('reason')}"
    assert v1["status"] == "ok"
    assert v1["max_norm_weight"] <= m.FINETUNE_NORM_WEIGHT_MAX
    assert v1["global_grad_norm"] <= m.FINETUNE_GRAD_NORM_MAX
    for cat, gn in (v1.get("param_group_grad_norms") or {}).items():
        assert gn <= m.FINETUNE_PARAM_GROUP_GRAD_MAX, f"group {cat} {gn}"
    print("[smoke] TEST 1 PASS")

    print("\n[smoke] === TEST 2: layer-1 norm pump ===")
    snapshot_norm = []
    pump = 1000.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n.lower() and n.endswith(".weight"):
                snapshot_norm.append((n, p.data.clone()))
                p.data.mul_(pump)
                if len(snapshot_norm) >= 4:
                    break  # one pumped layer is enough to trip L1
    assert snapshot_norm, "no norm weights found?!"
    print(f"[smoke] pumped {len(snapshot_norm)} norm tensors by {pump}x")
    v2 = m.trainability_probe(model)
    print(json.dumps(summarize(v2), default=str, indent=2))
    assert v2["ok"] is False
    assert v2["status"] == "anti_finetune"
    assert v2["reason"].startswith("norm_weight_cap"), v2["reason"]
    print("[smoke] TEST 2 PASS")
    # restore
    with torch.no_grad():
        for n, w in snapshot_norm:
            dict(model.named_parameters())[n].data.copy_(w)
    snapshot_norm.clear()

    print("\n[smoke] === TEST 3: post-restore honest king (snapshot integrity) ===")
    v3 = m.trainability_probe(model)
    print(json.dumps(summarize(v3), default=str, indent=2))
    assert v3["ok"] is True, f"post-restore should pass: {v3.get('reason')}"
    print("[smoke] TEST 3 PASS")

    print("\n[smoke] === TEST 4: param-group grad blow-up ===")
    # Pump every q_proj weight by 1e3 — Layer 1 untouched (those aren't norms),
    # forward stays finite for one batch, but the gradient on the attn group
    # explodes way past the per-group cap.
    snapshot_qp = []
    qp_pump = 1000.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "q_proj.weight" in n:
                snapshot_qp.append((n, p.data.clone()))
                p.data.mul_(qp_pump)
    assert snapshot_qp, "no q_proj.weight found?!"
    print(f"[smoke] pumped {len(snapshot_qp)} q_proj.weight tensors by {qp_pump}x")
    v4 = m.trainability_probe(model)
    print(json.dumps(summarize(v4), default=str, indent=2))
    assert v4["ok"] is False, "should fail somewhere on attn explosion"
    assert v4["status"] == "anti_finetune"
    # We accept either layer-3 (global) or layer-4 (per-group:attn) trip.
    assert ("global_grad_norm:" in v4["reason"]
            or "param_group_grad:attn" in v4["reason"]
            or "loss_non_finite" in v4["reason"]
            or "grad_non_finite" in v4["reason"]), v4["reason"]
    print("[smoke] TEST 4 PASS")
    with torch.no_grad():
        for n, w in snapshot_qp:
            dict(model.named_parameters())[n].data.copy_(w)
    snapshot_qp.clear()

    print("\n[smoke] === TEST 5: final restore re-probe ===")
    v5 = m.trainability_probe(model)
    print(json.dumps(summarize(v5), default=str, indent=2))
    assert v5["ok"] is True, f"final restore should pass: {v5.get('reason')}"
    print("[smoke] TEST 5 PASS")

    print("\n[smoke] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
