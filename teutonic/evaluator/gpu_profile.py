"""Explicit eight-GPU deployment profiles; included in scorer cache identities."""

import os

ENVIRONMENT_KEY = "TEUTONIC_EVALUATOR_GPU_PROFILE"


def gpu_profile():
    value = os.environ.get(ENVIRONMENT_KEY, "b300")
    if value not in {"b300", "h200"}:
        raise ValueError(f"{ENVIRONMENT_KEY} must be b300 or h200")
    return value


def mimo_tensor_parallel_size():
    return 2 if gpu_profile() == "h200" else 1


def parallel_strategy():
    return "tp2_replicas" if gpu_profile() == "h200" else "replicas_with_large_document_tp2"
