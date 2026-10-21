import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID, MASKED_POLICY_VERSION
from teutonic.evaluation.policy import apply_specialist_source_gate, paired_bootstrap_verdict


def sources(own):
    return [
        {"name": name, "proportion": 0.7 if name == own else 0.15}
        for name in ("math", "code", "text")
    ]


def engine_verdict(own, differences, labels, counts, threshold=0.0014, legacy=True):
    # Exercise the production verdict adapter without importing CUDA dependencies.
    path = Path(__file__).parents[2] / "teutonic/evaluator/engine.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "bootstrap_verdict")
    namespace = {
        "EvalRequest": SimpleNamespace,
        "paired_bootstrap_verdict": paired_bootstrap_verdict,
        "apply_specialist_source_gate": apply_specialist_source_gate,
        "MASKED_POLICY_VERSION": MASKED_POLICY_VERSION,
        "DOCUMENT_EOS_TOKEN_ID": DOCUMENT_EOS_TOKEN_ID,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    request = SimpleNamespace(
        bootstrap_seed=0, n_bootstrap=200, alpha=0.1,
        delta_threshold=threshold, long_documents={"enabled": True},
        dataset_sources=sources(own) if own else [{"name": "main-data", "proportion": 1}],
    )
    result = namespace["bootstrap_verdict"](
        differences, [0.0] * len(differences), request, counts, labels,
    )
    if legacy:
        apply_specialist_source_gate(result, differences, [0.0] * len(differences), counts,
                                     labels, request.dataset_sources, threshold)
    return result


@pytest.mark.parametrize("own", ["math", "code", "text"])
def test_other_sources_cannot_compensate_for_weak_specialist(own):
    labels = [own] * 70 + [s["name"] for s in sources(own) if s["name"] != own] * 15
    result = engine_verdict(own, [0.0013] * 70 + [0.1] * 30, labels, [100] * 100)
    assert result["lcb"] > 0.0014
    assert result["accepted"] is False
    assert result["verdict"] == "king"
    assert result["specialist_source_gate"]["source"] == own
    assert result["specialist_source_gate"]["passed"] is False


@pytest.mark.parametrize("advantage, passed", [(0.00139999, False), (0.0014, True)])
def test_source_floor_is_inclusive_and_uses_unrounded_losses(advantage, passed):
    result = engine_verdict(
        "math", [advantage, 0.1, 0.1], ["math", "code", "text"], [1, 100, 100],
    )
    assert result["lcb"] > 0.0014
    assert result["accepted"] is passed
    assert result["specialist_source_gate"]["passed"] is passed


def test_windows_and_documents_are_combined_by_scored_tokens():
    # A short window improves; a longer document regresses. Equal-unit averaging
    # would pass, but token-weighted math improvement is negative.
    result = engine_verdict(
        "math", [0.01, -0.001, 0.1, 0.1],
        ["math", "math", "code", "text"], [100, 8000, 10000, 10000],
    )
    gate = result["specialist_source_gate"]
    assert gate["n_scored_tokens"] == 8100
    assert gate["mu_hat"] == pytest.approx(-7 / 8100)
    assert result["accepted"] is False


def test_source_pass_does_not_override_failed_combined_lcb():
    result = engine_verdict("math", [0.004, -0.1, -0.1], ["math", "code", "text"], [1] * 3,
                            threshold=0.003)
    assert result["specialist_source_gate"]["passed"] is True
    assert result["accepted"] is False


def test_main_has_no_source_gate():
    result = engine_verdict(None, [0.002] * 3, None, [1] * 3)
    assert result["accepted"] is True
    assert "specialist_source_gate" not in result


def test_missing_source_in_evaluated_prefix_cannot_pass():
    result = engine_verdict("math", [0.1, 0.1], ["code", "text"], [1, 1])
    assert result["accepted"] is False
    assert result["specialist_source_gate"]["mu_hat"] is None


@pytest.mark.parametrize("labels", [None, ["math"], ["math", "code", "unknown"]])
def test_missing_or_invalid_source_labels_fail_closed(labels):
    with pytest.raises(ValueError, match="source"):
        engine_verdict("math", [0.1] * 3, labels, [1] * 3)


@pytest.mark.parametrize("threshold, passed", [(0.001, True), (0.003, False)])
def test_source_floor_tracks_configured_domain_threshold(threshold, passed):
    result = engine_verdict(
        "code", [0.002, 0.1, 0.1], ["code", "math", "text"], [1, 100, 100],
        threshold=threshold,
    )
    assert result["lcb"] > threshold
    assert result["specialist_source_gate"]["min_mu_hat"] == threshold
    assert result["specialist_source_gate"]["passed"] is passed
    assert result["accepted"] is passed


def test_category_policy_accepts_whole_mix_without_largest_source_floor():
    result = engine_verdict("math", [.0013] * 70 + [.1] * 30,
                            ["math"] * 70 + ["code"] * 15 + ["text"] * 15,
                            [100] * 100, legacy=False)
    assert result["accepted"] is True
    assert "specialist_source_gate" not in result


def test_category_policy_requires_strict_whole_mix_lcb():
    result = engine_verdict(None, [.002] * 10, None, [100] * 10,
                            threshold=.002, legacy=False)
    assert result["accepted"] is False
