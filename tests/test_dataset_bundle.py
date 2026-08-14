"""Dataset-bundle resolution: the eval mixture must never change silently.

`delta_threshold` is a fixed constant, so the mixture the losses are computed
over is part of the scoring contract. These tests pin the two properties that
matter: an unfetchable bundle raises rather than substituting a fallback, and a
bundle whose bytes drift from chain.toml's pin is refused.
"""
from __future__ import annotations

import json

import pytest

import npy_sources as ns


@pytest.fixture(autouse=True)
def _clean_bundle_state():
    """Each test starts with an unresolved module (RESOLVED_BUNDLE is global)."""
    saved_urls = list(ns.DEFAULT_MANIFEST_URLS)
    saved_weights = dict(ns.DEFAULT_SOURCE_WEIGHT_MAP)
    saved_resolved = dict(ns.RESOLVED_BUNDLE)
    ns.RESOLVED_BUNDLE.clear()
    yield
    ns.DEFAULT_MANIFEST_URLS[:] = saved_urls
    ns.DEFAULT_SOURCE_WEIGHT_MAP.clear()
    ns.DEFAULT_SOURCE_WEIGHT_MAP.update(saved_weights)
    ns.RESOLVED_BUNDLE.clear()
    ns.RESOLVED_BUNDLE.update(saved_resolved)


BUNDLE = {
    "sources": [
        {"name": "alpha", "manifest_url": "https://x/dataset/alpha/manifest.json", "weight": 0.7},
        {"name": "beta", "manifest_url": "https://x/dataset/beta/manifest.json", "weight": 0.3},
        {
            "name": "disabled",
            "manifest_url": "https://x/dataset/disabled/manifest.json",
            "weight": 1.0,
            "enabled": False,
        },
    ]
}
BUNDLE_RAW = json.dumps(BUNDLE).encode()


def _fake_urlopen(raw: bytes):
    class _Resp:
        def read(self):
            return raw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda *a, **kw: _Resp()


# --------------------------------------------------------------------------
# bundle_to_sources
# --------------------------------------------------------------------------


def test_disabled_sources_are_dropped():
    urls, weights = ns.bundle_to_sources(BUNDLE)
    assert urls == [
        "https://x/dataset/alpha/manifest.json",
        "https://x/dataset/beta/manifest.json",
    ]
    assert weights["alpha"] == 0.7
    assert weights["beta"] == 0.3
    assert "disabled" not in weights


def test_entries_missing_url_or_name_are_skipped():
    urls, weights = ns.bundle_to_sources(
        {"sources": [{"name": "no-url", "weight": 1.0}, {"manifest_url": "https://x/y.json"}]}
    )
    assert urls == []
    assert weights == {}


def test_weight_map_aliases_url_directory_name():
    """Bundle names may use hyphens where the URL path uses underscores."""
    _, weights = ns.bundle_to_sources(
        {
            "sources": [
                {
                    "name": "dolma3-longmino",
                    "manifest_url": "https://x/dataset/dolma3_longmino_pool_8k/manifest.json",
                    "weight": 0.1,
                }
            ]
        }
    )
    assert weights["dolma3-longmino"] == 0.1
    assert weights["dolma3_longmino_pool_8k"] == 0.1


# --------------------------------------------------------------------------
# fetch + pin
# --------------------------------------------------------------------------


def test_digest_is_over_raw_bytes():
    assert ns.bundle_digest(b"{}") == (
        "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_fetch_returns_parsed_manifest_and_digest(monkeypatch):
    monkeypatch.setattr(ns, "urlopen", _fake_urlopen(BUNDLE_RAW))
    bundle, digest = ns.fetch_bundle_manifest("https://x/bundle.json")
    assert bundle == BUNDLE
    assert digest == ns.bundle_digest(BUNDLE_RAW)


def test_fetch_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection reset")
        return _fake_urlopen(BUNDLE_RAW)()

    monkeypatch.setattr(ns, "urlopen", flaky)
    monkeypatch.setattr(ns.time, "sleep", lambda _s: None)
    bundle, _ = ns.fetch_bundle_manifest("https://x/bundle.json", attempts=3, backoff_s=0)
    assert bundle == BUNDLE
    assert calls["n"] == 3


def test_fetch_raises_after_exhausting_attempts(monkeypatch):
    """The regression this suite exists for: no silent fallback mixture."""

    def always_fails(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr(ns, "urlopen", always_fails)
    monkeypatch.setattr(ns.time, "sleep", lambda _s: None)
    with pytest.raises(ns.BundleManifestError):
        ns.fetch_bundle_manifest("https://x/bundle.json", attempts=2, backoff_s=0)


def test_malformed_json_raises(monkeypatch):
    monkeypatch.setattr(ns, "urlopen", _fake_urlopen(b"not json"))
    monkeypatch.setattr(ns.time, "sleep", lambda _s: None)
    with pytest.raises(ns.BundleManifestError):
        ns.fetch_bundle_manifest("https://x/bundle.json", attempts=1, backoff_s=0)


def test_unpinned_bundle_is_allowed():
    ns.verify_bundle_digest("sha256:" + "a" * 64, "")


def test_pinned_bundle_accepts_exact_match():
    ns.verify_bundle_digest("sha256:" + "a" * 64, "sha256:" + "a" * 64)


def test_pinned_bundle_rejects_drift():
    with pytest.raises(ns.BundleManifestError, match="digest mismatch"):
        ns.verify_bundle_digest("sha256:" + "b" * 64, "sha256:" + "a" * 64)


def test_bundle_with_no_enabled_sources_raises(monkeypatch):
    raw = json.dumps({"sources": [{"name": "a", "manifest_url": "u", "enabled": False}]}).encode()
    monkeypatch.setattr(ns, "urlopen", _fake_urlopen(raw))
    with pytest.raises(ns.BundleManifestError, match="no enabled sources"):
        ns.resolve_bundle_sources("https://x/bundle.json", "")


# --------------------------------------------------------------------------
# ensure_bundle_resolved
# --------------------------------------------------------------------------


def test_resolve_populates_defaults_in_place(monkeypatch):
    """Mutation, not rebinding: eval_server_two_sources imports these by value."""
    monkeypatch.setattr(ns, "urlopen", _fake_urlopen(BUNDLE_RAW))
    monkeypatch.setattr(ns, "_raw_manifest_urls", "")
    monkeypatch.setattr(ns, "_raw_weight_map", "")
    urls_obj = ns.DEFAULT_MANIFEST_URLS
    weights_obj = ns.DEFAULT_SOURCE_WEIGHT_MAP

    resolved = ns.ensure_bundle_resolved(force=True)

    assert ns.DEFAULT_MANIFEST_URLS is urls_obj
    assert ns.DEFAULT_SOURCE_WEIGHT_MAP is weights_obj
    assert len(urls_obj) == 2
    assert weights_obj["alpha"] == 0.7
    assert resolved["origin"] == "bundle"
    assert resolved["digest"] == ns.bundle_digest(BUNDLE_RAW)


def test_resolve_is_idempotent(monkeypatch):
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return _fake_urlopen(BUNDLE_RAW)()

    monkeypatch.setattr(ns, "urlopen", counting)
    monkeypatch.setattr(ns, "_raw_manifest_urls", "")
    monkeypatch.setattr(ns, "_raw_weight_map", "")
    ns.ensure_bundle_resolved(force=True)
    ns.ensure_bundle_resolved()
    assert calls["n"] == 1


def test_env_override_skips_the_network(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("network must not be touched when env override is set")

    monkeypatch.setattr(ns, "urlopen", explode)
    monkeypatch.setattr(ns, "_raw_manifest_urls", "https://local/manifest.json")
    resolved = ns.ensure_bundle_resolved(force=True)
    assert resolved["origin"] == "env_override"
    assert resolved["pinned"] is False


def test_resolve_failure_propagates(monkeypatch):
    """Startup must abort, not fall through to an unknown mixture."""

    def always_fails(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr(ns, "urlopen", always_fails)
    monkeypatch.setattr(ns, "time", type("T", (), {"sleep": staticmethod(lambda _s: None)}))
    monkeypatch.setattr(ns, "_raw_manifest_urls", "")
    monkeypatch.setattr(ns, "_raw_weight_map", "")
    monkeypatch.setattr(ns, "BUNDLE_FETCH_ATTEMPTS", 1)
    with pytest.raises(ns.BundleManifestError):
        ns.ensure_bundle_resolved(force=True)
    assert ns.RESOLVED_BUNDLE == {}


def test_default_sources_refuses_empty_mixture():
    req = ns.MultiSourceEvalRequest(
        king_repo="a/b", challenger_repo="c/d", npy_manifests=[], npy_sources=[]
    )
    with pytest.raises(ns.BundleManifestError):
        ns.default_sources(req)
