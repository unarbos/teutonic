#!/usr/bin/env python3
"""Compare network loss curve vs local training loss curve.

Pulls the network's king-change history from dashboard.json and the local
training log from R2 (or local file), then prints a side-by-side comparison
showing loss vs elapsed time for both.

Usage:
    python compare_curves.py
    python compare_curves.py --local-log /tmp/experiment/full_run/train_log.jsonl
"""
import argparse
import json
import os
import sys
from datetime import datetime

import boto3
from botocore.config import Config as BotoConfig


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
        aws_access_key_id=os.environ["TEUTONIC_R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TEUTONIC_R2_SECRET_KEY"],
        region_name="auto",
        config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
    )


def parse_time(ts):
    ts = ts.replace("+00:00", "").replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def get_network_curve(client, bucket):
    """Extract the network's loss-over-time from dashboard history."""
    raw = client.get_object(Bucket=bucket, Key="dashboard.json")["Body"].read()
    dashboard = json.loads(raw)

    points = []
    for h in dashboard.get("history", []):
        v = h.get("verdict")
        if not v or not isinstance(v, dict):
            continue
        ts = h.get("started_at") or h.get("timestamp") or v.get("timestamp")
        if not ts:
            continue
        t = parse_time(ts)
        if not t:
            continue

        king_loss = v.get("avg_king_loss")
        chall_loss = v.get("avg_challenger_loss")
        accepted = v.get("accepted", False)

        if king_loss is not None:
            points.append({
                "time": t,
                "loss": chall_loss if accepted else king_loss,
                "accepted": accepted,
                "challenger_loss": chall_loss,
                "king_loss": king_loss,
                "id": h.get("challenge_id", ""),
                "repo": h.get("challenger_repo", ""),
            })

    points.sort(key=lambda x: x["time"])

    # Build the king loss trajectory: start with first eval's king, then track accepted changes
    curve = []
    if points:
        t0 = points[0]["time"]
        current_loss = points[0]["king_loss"]
        curve.append({"hours": 0, "loss": current_loss, "event": "start"})

        for p in points:
            if p["accepted"]:
                hours = (p["time"] - t0).total_seconds() / 3600
                current_loss = p["challenger_loss"]
                curve.append({
                    "hours": round(hours, 2),
                    "loss": round(current_loss, 6),
                    "event": f"king_change ({p['id']})",
                    "repo": p["repo"],
                })

    return curve, points[0]["time"] if points else None


def get_local_curve(source, client=None, bucket=None):
    """Load training log from local file or R2."""
    lines = []
    if os.path.exists(source):
        with open(source) as f:
            lines = f.readlines()
    elif client and bucket:
        try:
            raw = client.get_object(Bucket=bucket, Key=source)["Body"].read()
            lines = raw.decode().strip().split("\n")
        except Exception as e:
            print(f"Could not fetch {source} from R2: {e}")
            return [], None

    if not lines:
        return [], None

    curve = []
    t0 = None
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        if d.get("event") == "start":
            t0_str = d.get("timestamp")
            if t0_str:
                t0 = parse_time(t0_str)
            init_loss = d.get("init_val_loss")
            if init_loss:
                curve.append({"hours": 0, "loss": round(init_loss, 6), "event": "start"})

        elif d.get("event") == "validation":
            ts = parse_time(d["timestamp"]) if d.get("timestamp") else None
            hours = (ts - t0).total_seconds() / 3600 if ts and t0 else d.get("elapsed_s", 0) / 3600
            curve.append({
                "hours": round(hours, 2),
                "loss": d["val_loss"],
                "event": f"val (step {d.get('step', '?')})",
                "tokens_B": d.get("total_tokens_B"),
            })

        elif d.get("event") == "step" and d.get("step", 0) % 100 == 0:
            ts = parse_time(d["timestamp"]) if d.get("timestamp") else None
            hours = (ts - t0).total_seconds() / 3600 if ts and t0 else d.get("elapsed_s", 0) / 3600
            curve.append({
                "hours": round(hours, 2),
                "loss": d["loss"],
                "event": f"train (step {d['step']})",
                "tokens_B": d.get("total_tokens_B"),
                "tok_per_sec": d.get("tokens_per_sec"),
            })

    return curve, t0


def print_comparison(net_curve, local_curve):
    print()
    print("=" * 90)
    print("LOSS CURVE COMPARISON: Network vs Local 8xB200 Training")
    print("=" * 90)

    print()
    print("--- Network (King Loss Over Time) ---")
    print(f"{'Hours':>8}  {'Loss':>10}  Event")
    print("-" * 60)
    for p in net_curve:
        print(f"{p['hours']:>8.2f}  {p['loss']:>10.6f}  {p['event']}")

    print()
    print("--- Local 8xB200 Training ---")
    if not local_curve:
        print("  (no data yet -- training may not have started)")
    else:
        print(f"{'Hours':>8}  {'Loss':>10}  {'Tokens':>8}  {'Tok/s':>10}  Event")
        print("-" * 80)
        for p in local_curve:
            tok = f"{p.get('tokens_B', 0):.1f}B" if p.get('tokens_B') else ""
            tps = f"{p.get('tok_per_sec', 0):.0f}" if p.get('tok_per_sec') else ""
            print(f"{p['hours']:>8.2f}  {p['loss']:>10.6f}  {tok:>8}  {tps:>10}  {p['event']}")

    # Side by side at matching time points
    if net_curve and local_curve and len(local_curve) > 1:
        print()
        print("--- Side-by-Side at Matching Hours ---")
        print(f"{'Hours':>8}  {'Network':>12}  {'Local':>12}  {'Delta':>10}  {'Winner':>10}")
        print("-" * 65)

        local_times = [(p["hours"], p["loss"]) for p in local_curve]

        for np_ in net_curve:
            h = np_["hours"]
            # Find closest local point
            closest = min(local_times, key=lambda x: abs(x[0] - h))
            if abs(closest[0] - h) < 0.5:
                delta = np_["loss"] - closest[1]
                winner = "LOCAL" if closest[1] < np_["loss"] else "NETWORK"
                print(f"{h:>8.2f}  {np_['loss']:>12.6f}  {closest[1]:>12.6f}  {delta:>+10.6f}  {winner:>10}")

    print()
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-log", default="experiments/baseline_train_log.jsonl",
                        help="Path to local train log (file or R2 key)")
    args = parser.parse_args()

    client = get_r2_client()
    bucket = os.environ.get("TEUTONIC_R2_BUCKET", "constantinople")

    print("Fetching network curve from dashboard...")
    net_curve, net_start = get_network_curve(client, bucket)
    print(f"  Found {len(net_curve)} data points, network started at {net_start}")

    print("Fetching local training curve...")
    local_curve, local_start = get_local_curve(args.local_log, client, bucket)
    print(f"  Found {len(local_curve)} data points")

    print_comparison(net_curve, local_curve)


if __name__ == "__main__":
    main()
