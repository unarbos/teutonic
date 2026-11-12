import copy

import pytest

from scripts.build_website_dataset_inventory import inventory


def manifest():
    return {
        "format": "teutonic-packed-text-v2", "documents": 3, "utf8_bytes": 30,
        "categories": [{"dataset": "code", "category": "reasoning",
                        "documents": 3, "utf8_bytes": 30, "original_mimo_tokens": 17}],
        "objects": [
            {"dataset": "code", "category": "reasoning", "role": "text",
             "document_count": 2, "size_bytes": 20},
            {"dataset": "code", "category": "reasoning", "role": "text",
             "document_count": 1, "size_bytes": 10},
            {"dataset": "code", "category": "reasoning", "role": "index",
             "document_count": 3, "size_bytes": 500},
        ],
    }


def test_inventory_uses_original_tokens_and_counts_only_text_shards():
    pin = {"bucket": "dataset", "key": "teutonic-text/manifest.json", "sha256": "a" * 64}
    result = inventory(manifest(), pin)
    source = result["sources"][0]
    assert result["dataset"] == pin
    assert result["token_count_basis"] == "original_mimo_token_count"
    assert source["total_tokens"] == 17
    assert source["total_shards"] == 2
    assert source["total_documents"] == 3
    assert source["total_utf8_bytes"] == 30


@pytest.mark.parametrize("change", ["category", "dataset", "duplicate", "missing"])
def test_inventory_rejects_inconsistent_totals(change):
    data = manifest()
    if change == "category":
        data["categories"][0]["documents"] += 1
    elif change == "dataset":
        data["utf8_bytes"] += 1
    elif change == "duplicate":
        data["categories"].append(copy.deepcopy(data["categories"][0]))
    else:
        data["categories"] = []
    with pytest.raises(ValueError):
        inventory(data, {"bucket": "dataset", "key": "teutonic-text/manifest.json",
                         "sha256": "a" * 64})
