#!/usr/bin/env python3
"""Analyze per-parameter L2 norms across king and challenger models.

Downloads safetensors index/files from HuggingFace and computes norms
without loading full models onto GPU — uses safetensors metadata + CPU tensors.
"""
import json
import sys
import os

from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_file
import torch


def get_norm_profile(repo, revision=None, label=None):
    """Download safetensors and compute per-parameter L2 norms on CPU."""
    label = label or repo
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  repo: {repo}  rev: {(revision or 'main')[:12]}")
    print(f"{'='*70}")

    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    try:
        files = api.list_repo_files(repo, revision=revision, token=token)
    except Exception as e:
        print(f"  ERROR listing repo: {e}")
        return None

    st_files = sorted(f for f in files if f.endswith(".safetensors"))
    if not st_files:
        print("  No safetensors files found")
        return None

    norms = {}
    for fname in st_files:
        try:
            path = hf_hub_download(repo, fname, revision=revision, token=token)
        except Exception as e:
            print(f"  ERROR downloading {fname}: {e}")
            continue

        tensors = load_file(path, device="cpu")
        for name, tensor in tensors.items():
            t = tensor.float()
            norm = torch.linalg.vector_norm(t).item()
            numel = t.numel()
            has_nan = not torch.isfinite(t).all()
            norms[name] = {
                "norm": norm,
                "numel": numel,
                "mean_abs": t.abs().mean().item(),
                "max_abs": t.abs().max().item(),
                "has_nan": bool(has_nan),
            }
        del tensors

    return norms


def compare_norms(king_norms, challenger_norms, label):
    """Compare challenger norms vs king and report ratios."""
    print(f"\n--- Comparison: {label} vs King ---")

    ratios = []
    violations_5x = []
    violations_10x = []
    nan_params = []

    for name in sorted(king_norms.keys()):
        if name not in challenger_norms:
            continue

        k = king_norms[name]
        c = challenger_norms[name]

        if c["has_nan"]:
            nan_params.append(name)
            continue

        if k["norm"] < 1e-8:
            continue

        ratio = c["norm"] / k["norm"]
        ratios.append((name, ratio, k["norm"], c["norm"]))

        if ratio > 10.0:
            violations_10x.append((name, ratio, k["norm"], c["norm"]))
        elif ratio > 5.0:
            violations_5x.append((name, ratio, k["norm"], c["norm"]))

    if not ratios:
        print("  No comparable parameters found")
        return {}

    all_ratios = [r[1] for r in ratios]
    print(f"  Parameters compared: {len(ratios)}")
    print(f"  NaN/Inf params:      {len(nan_params)}")
    print(f"  Ratio stats:")
    print(f"    min:    {min(all_ratios):.6f}")
    print(f"    max:    {max(all_ratios):.6f}")
    print(f"    mean:   {sum(all_ratios)/len(all_ratios):.6f}")
    print(f"    median: {sorted(all_ratios)[len(all_ratios)//2]:.6f}")

    p95 = sorted(all_ratios)[int(len(all_ratios) * 0.95)]
    p99 = sorted(all_ratios)[int(len(all_ratios) * 0.99)]
    print(f"    p95:    {p95:.6f}")
    print(f"    p99:    {p99:.6f}")

    print(f"  Violations >5x:  {len(violations_5x)}")
    print(f"  Violations >10x: {len(violations_10x)}")

    if violations_10x:
        print(f"\n  TOP 10x+ violations:")
        for name, ratio, kn, cn in sorted(violations_10x, key=lambda x: -x[1])[:10]:
            print(f"    {ratio:8.2f}x  king={kn:.4f}  chall={cn:.4f}  {name}")

    if violations_5x:
        print(f"\n  5x-10x violations:")
        for name, ratio, kn, cn in sorted(violations_5x, key=lambda x: -x[1])[:10]:
            print(f"    {ratio:8.2f}x  king={kn:.4f}  chall={cn:.4f}  {name}")

    if nan_params:
        print(f"\n  NaN/Inf parameters:")
        for name in nan_params[:5]:
            print(f"    {name}")

    # Show top-10 highest absolute norms in challenger
    top_abs = sorted(
        [(name, challenger_norms[name]["norm"]) for name in challenger_norms],
        key=lambda x: -x[1]
    )[:10]
    print(f"\n  Top 10 absolute norms (challenger):")
    for name, norm in top_abs:
        print(f"    {norm:12.2f}  {name}")

    return {
        "label": label,
        "n_params": len(ratios),
        "n_nan": len(nan_params),
        "ratio_min": min(all_ratios),
        "ratio_max": max(all_ratios),
        "ratio_mean": sum(all_ratios) / len(all_ratios),
        "ratio_p95": p95,
        "ratio_p99": p99,
        "violations_5x": len(violations_5x),
        "violations_10x": len(violations_10x),
    }


def main():
    king_repo = "unconst/Teutonic-I"
    king_rev = "1015e501ecb66f6d204c4eb2e3b71aaab9ff8851"

    challengers = [
        ("dasLOL/Teutonic-I-v1", None),
        ("dasLOL/Teutonic-I-v2", None),
        ("datlab755/Teutonic-I-merlin-85", None),
        ("datlab755/Teutonic-I-v207", None),
        ("ClarenceDan/Teutonic-I-Clarence-A5507", None),
        ("knsimon/Teutonic-I-50", None),
        ("tech-dev-ai/Teutonic-I-800", None),
        ("gtensorapp/Teutonic-I-v8a", None),
        ("kt3202/Teutonic-I-test-11", None),
        ("unconst/Teutonic-I-h5", None),
        ("s0wa48/Teutonic-I-iter4-marginal-1776214256", None),
        ("iotaminer/Teutonic-I-fwlr5e6-live-001", None),
    ]

    print("Analyzing king model...")
    king_norms = get_norm_profile(king_repo, revision=king_rev, label="KING")
    if king_norms is None:
        print("Failed to load king model norms")
        sys.exit(1)

    print(f"\nKing has {len(king_norms)} parameters")
    top_king = sorted(
        [(name, info["norm"]) for name, info in king_norms.items()],
        key=lambda x: -x[1]
    )[:10]
    print("Top 10 king norms:")
    for name, norm in top_king:
        print(f"  {norm:12.2f}  {name}")

    results = []
    for repo, rev in challengers:
        try:
            c_norms = get_norm_profile(repo, revision=rev, label=repo)
            if c_norms is None:
                continue
            stats = compare_norms(king_norms, c_norms, repo)
            if stats:
                results.append(stats)
            del c_norms
        except Exception as e:
            print(f"\nERROR processing {repo}: {e}")
            continue

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Repo':<55} {'max_ratio':>10} {'p99':>8} {'p95':>8} {'>5x':>5} {'>10x':>5} {'NaN':>5}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x["ratio_max"]):
        print(f"{r['label']:<55} {r['ratio_max']:>10.4f} {r['ratio_p99']:>8.4f} {r['ratio_p95']:>8.4f} {r['violations_5x']:>5} {r['violations_10x']:>5} {r['n_nan']:>5}")

    print("\n--- Threshold Recommendations ---")
    all_maxes = [r["ratio_max"] for r in results if r["n_nan"] == 0 and r["violations_10x"] == 0]
    if all_maxes:
        safe_max = max(all_maxes)
        print(f"Max ratio among clean models: {safe_max:.4f}")
        print(f"Suggested TEUTONIC_NORM_MULTIPLIER: {max(safe_max * 2, 5.0):.1f}")

    with open("/home/const/workspace/teutonic/scripts/norm_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to scripts/norm_analysis.json")


if __name__ == "__main__":
    main()
