#!/usr/bin/env python3
"""Packed-text configuration staging (local SQL only).

Historical token-shard administration lives under scripts/legacy/ for audit.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage_packed_configuration import main

if __name__ == "__main__":
    main()
