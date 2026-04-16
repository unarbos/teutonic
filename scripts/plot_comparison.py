#!/usr/bin/env python3
"""Generate comparison chart: Network king loss vs Local 8xB200 training loss over time."""
import json
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_time(ts):
    ts = ts.replace("+00:00", "").replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def load_network_curve(eval_history_path):
    with open(eval_history_path) as f:
        evals = json.loads(f.read())

    if not evals:
        return [], []

    t0 = parse_time(evals[0]["time"])

    king_loss_over_time = []
    current_king_loss = evals[0]["king_loss"]

    king_loss_over_time.append((0.0, current_king_loss))

    for e in evals:
        t = parse_time(e["time"])
        hours = (t - t0).total_seconds() / 3600
        if e["accepted"]:
            current_king_loss = e["chall_loss"]
            king_loss_over_time.append((hours, current_king_loss))

    hours = [p[0] for p in king_loss_over_time]
    losses = [p[1] for p in king_loss_over_time]
    return hours, losses


def load_local_curve(train_log_path):
    hours_train = []
    losses_train = []
    hours_val = []
    losses_val = []

    t0 = None
    with open(train_log_path) as f:
        for line in f:
            d = json.loads(line)
            ts = parse_time(d.get("timestamp", ""))

            if d.get("event") == "start":
                t0 = ts
                init_loss = d.get("init_val_loss")
                if init_loss:
                    hours_val.append(0.0)
                    losses_val.append(init_loss)

            elif d.get("event") == "step":
                h = d.get("elapsed_s", 0) / 3600
                hours_train.append(h)
                losses_train.append(d["loss"])

            elif d.get("event") == "validation":
                if ts and t0:
                    h = (ts - t0).total_seconds() / 3600
                else:
                    h = d.get("elapsed_s", 0) / 3600 if d.get("elapsed_s") else 0
                hours_val.append(h)
                losses_val.append(d["val_loss"])

    return hours_train, losses_train, hours_val, losses_val


def main():
    eval_history = sys.argv[1] if len(sys.argv) > 1 else "eval_history.json"
    train_log = sys.argv[2] if len(sys.argv) > 2 else "train_log_latest.jsonl"
    output = sys.argv[3] if len(sys.argv) > 3 else "comparison_chart.png"

    net_hours, net_losses = load_network_curve(eval_history)
    train_hours, train_losses, val_hours, val_losses = load_local_curve(train_log)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Network king loss (step function -- stays at each level until dethroned)
    if net_hours:
        # Extend to current time for step plot
        extended_hours = []
        extended_losses = []
        for i, (h, l) in enumerate(zip(net_hours, net_losses)):
            extended_hours.append(h)
            extended_losses.append(l)
            if i + 1 < len(net_hours):
                extended_hours.append(net_hours[i + 1])
                extended_losses.append(l)

        ax.plot(extended_hours, extended_losses, 'r-', linewidth=2.5, label='Network (King Loss)', zorder=3)
        ax.scatter(net_hours, net_losses, color='red', s=80, zorder=4, edgecolors='darkred', linewidth=1.5)

    # Local training loss (smoothed)
    if train_hours:
        ax.plot(train_hours, train_losses, color='#4488cc', alpha=0.3, linewidth=0.8, label='Local 8xB200 (Train Loss)')

        # Smoothed version
        window = min(20, len(train_losses) // 3)
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(train_losses, kernel, mode='valid')
            smoothed_hours = train_hours[window // 2 : window // 2 + len(smoothed)]
            ax.plot(smoothed_hours, smoothed, color='#2266aa', linewidth=2, label='Local 8xB200 (Train Loss, smoothed)')

    # Local validation loss
    if val_hours:
        ax.plot(val_hours, val_losses, 'b-o', linewidth=2.5, markersize=8, label='Local 8xB200 (Val Loss)',
                color='#1144aa', zorder=5, markeredgecolor='navy', markeredgewidth=1.5)

    ax.set_xlabel('Hours Since Start', fontsize=14)
    ax.set_ylabel('Loss (nats/token)', fontsize=14)
    ax.set_title('Teutonic SN3: Decentralized Network vs Centralized 8xB200 Training\n'
                 'Loss Over Time (lower is better)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.5)

    # Add annotations
    if net_losses:
        ax.annotate(f'Network best: {net_losses[-1]:.3f}',
                    xy=(net_hours[-1], net_losses[-1]),
                    xytext=(net_hours[-1] - 2, net_losses[-1] + 0.5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=11, color='red', fontweight='bold')

    if val_losses and len(val_losses) > 1:
        best_val = min(val_losses[1:])
        best_idx = val_losses.index(best_val)
        ax.annotate(f'Local best: {best_val:.3f}',
                    xy=(val_hours[best_idx], best_val),
                    xytext=(val_hours[best_idx] + 0.1, best_val + 0.3),
                    arrowprops=dict(arrowstyle='->', color='navy'),
                    fontsize=11, color='navy', fontweight='bold')

    # Add text box with key stats
    if val_losses and net_losses:
        stats_text = (
            f'Network: {len(net_hours)-1} king changes over {net_hours[-1]:.1f}h\n'
            f'  Start: {net_losses[0]:.2f} → Best: {net_losses[-1]:.3f}\n'
            f'Local 8xB200: {max(val_hours):.2f}h elapsed\n'
            f'  Start: {val_losses[0]:.2f} → Best: {min(val_losses[1:]):.3f}\n'
            f'  Throughput: ~310K tok/s'
        )
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Chart saved to {output}")


if __name__ == "__main__":
    main()
