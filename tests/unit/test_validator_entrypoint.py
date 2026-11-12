"""Exercise the preflight wired into the production validator entry point."""

import asyncio

import pytest
from test_document_protocol import request

from scripts.validator_service import contract_preflight
from teutonic.evaluation.protocol_v3 import ProtocolValidationError


def test_production_preflight_accepts_packed_document_request():
    assert asyncio.run(contract_preflight(request())) is None


@pytest.mark.parametrize("invalid", ["legacy", "missing_dataset_pin", "invalid_threshold"])
def test_production_preflight_rejects_invalid_request(invalid):
    payload = request()
    if invalid == "legacy":
        payload["protocol_version"] = "teutonic-evaluator-v2"
    elif invalid == "missing_dataset_pin":
        del payload["dataset"]["dataset"]
    else:
        payload["limits"]["delta_threshold"] = float("nan")
    with pytest.raises(ProtocolValidationError):
        asyncio.run(contract_preflight(payload))
