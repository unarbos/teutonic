#!/usr/bin/env python3
import argparse
import json
import random


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample_take(items, n, rng):
    items = list(items)
    if n >= len(items):
        rng.shuffle(items)
        return items
    return rng.sample(items, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--general-frac", type=float, default=0.7)
    ap.add_argument("--hard-frac", type=float, default=0.2)
    ap.add_argument("--easy-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = list(read_jsonl(args.scores))

    general = [r for r in rows if r.get("bucket") == "general"]
    hard = [r for r in rows if r.get("bucket") == "hard"]
    easy = [r for r in rows if r.get("bucket") == "easy"]
    clean = [r for r in rows if r.get("bucket") != "suspicious"]
    suspicious = [r for r in rows if r.get("bucket") == "suspicious"]

    val = sample_take(clean, min(args.val_size, len(clean)), rng)
    val_ids = {r["sample_index"] for r in val}
    general = [r for r in general if r["sample_index"] not in val_ids]
    hard = [r for r in hard if r["sample_index"] not in val_ids]
    easy = [r for r in easy if r["sample_index"] not in val_ids]

    n_general = int(args.train_size * args.general_frac)
    n_hard = int(args.train_size * args.hard_frac)
    n_easy = args.train_size - n_general - n_hard

    train = []
    train.extend(sample_take(general, n_general, rng))
    train.extend(sample_take(hard, n_hard, rng))
    train.extend(sample_take(easy, n_easy, rng))
    rng.shuffle(train)

    with open(args.train_out, "w") as f:
        for r in train:
            f.write(json.dumps({"input_ids": r["input_ids"]}) + "\n")

    with open(args.val_out, "w") as f:
        for r in val:
            f.write(json.dumps({"input_ids": r["input_ids"]}) + "\n")

    print(json.dumps({
        "train": len(train),
        "val": len(val),
        "available": {
            "general": len(general),
            "hard": len(hard),
            "easy": len(easy),
            "suspicious": len(suspicious),
        }
    }, indent=2))


if __name__ == "__main__":
    main()
