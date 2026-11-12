#!/usr/bin/env python3
"""Packed-text manifest preparation (read-only R2, local output).

Historical token-shard administration lives under scripts/legacy/ for audit.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_packed_evaluation import main

if __name__ == "__main__":
    main()
