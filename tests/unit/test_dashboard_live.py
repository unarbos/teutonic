import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from teutonic.dashboard.contracts import DashboardContractError
from teutonic.dashboard.live import canonical_live_json, publish_live


def live_payload():
    return {
        "schema_version": 1, "generated_at": "2026-08-18T12:00:00Z",
        "netuid": 306, "generation": "test", "source_watermark": 7,
        "competitions": [{
            "key": "main", "name": "quasar", "queue": [], "current_eval": None,
            "service_status": {
                "overall": "offline", "validator_phase": None,
                "validator_heartbeat_age_seconds": None, "current_evaluation_age_seconds": None,
                "weight_delayed": False, "services": [],
            },
        }],
    }


def test_live_publication_uses_separate_object_and_short_cache():
    client = Mock()
    value = live_payload()
    size = publish_live(client, "dashboard", value)
    kwargs = client.put_object.call_args.kwargs
    assert kwargs["Key"] == "dashboard-live.json"
    assert kwargs["CacheControl"] == "public, max-age=5, must-revalidate"
    assert json.loads(kwargs["Body"]) == value
    assert size == len(kwargs["Body"])
    # No full-object read, history, market or read/modify/write race.
    client.get_object.assert_not_called()


@pytest.mark.parametrize("mutation", ["private_field", "duplicate", "no_main", "nan", "bad_current"])
def test_invalid_snapshot_is_not_uploaded(mutation):
    value = live_payload()
    item = value["competitions"][0]
    if mutation == "private_field":
        item["private_model_url"] = "DO-NOT-LEAK"
    elif mutation == "duplicate":
        value["competitions"].append(deepcopy(item))
    elif mutation == "no_main":
        item["key"] = "mix01"
    elif mutation == "nan":
        value["source_watermark"] = float("nan")
    else:
        item["current_eval"] = {"private_diagnostic_reference": "DO-NOT-LEAK"}
    client = Mock()
    with pytest.raises(DashboardContractError):
        publish_live(client, "dashboard", value)
    client.put_object.assert_not_called()


def test_canonical_snapshot_is_stable():
    assert canonical_live_json(live_payload()) == canonical_live_json(live_payload())
