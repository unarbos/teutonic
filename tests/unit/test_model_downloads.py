from contextlib import ExitStack
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from teutonic.evaluator.cache_lock import cache_lock
from teutonic.evaluator.model_downloads import ModelDownloads


@pytest.mark.parametrize("failed_role", ["king", "challenger"])
def test_download_failure_is_raised_by_both_check_and_gate(failed_role):
    calls = []

    def resolve(role):
        calls.append(role)
        if role == failed_role:
            raise ValueError("fixture integrity failure")
        return role + "-verified"

    with ModelDownloads(SimpleNamespace(resolve=resolve), "king", "challenger") as downloads:
        with pytest.raises(ValueError, match="integrity failure"):
            downloads.result()
        with pytest.raises(ValueError, match="integrity failure"):
            downloads.check()
        assert downloads.status()[failed_role]["state"] == "failed"
    assert calls == (["king"] if failed_role == "king" else ["king", "challenger"])


def test_reference_failure_joins_downloader_before_cache_cleanup(tmp_path):
    started, finish, returned = Event(), Event(), Event()
    downloads = []
    calls, errors = [], []

    def resolve(role):
        calls.append(role)
        started.set()
        assert finish.wait(5)
        with cache_lock(tmp_path, cleanup=True) as free:
            assert not free
        return str(tmp_path / role)

    def evaluation():
        try:
            with ExitStack() as stack:
                stack.enter_context(cache_lock(tmp_path))
                pending = stack.enter_context(ModelDownloads(SimpleNamespace(resolve=resolve), "king", "challenger"))
                downloads.append(pending)
                assert started.wait(5)
                raise RuntimeError("reference failed")
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            returned.set()

    thread = Thread(target=evaluation)
    thread.start()
    try:
        assert started.wait(5)
        # Wait for the exit path, not a timing-dependent sleep. The writer must
        # still be joined before the surrounding cache-lock context can exit.
        while not downloads:
            assert not returned.wait(0.01)
        assert downloads[0]._stop.wait(5)
        assert not returned.is_set()
        with cache_lock(tmp_path, cleanup=True) as free:
            assert not free
    finally:
        finish.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert returned.is_set()
    assert errors == ["reference failed"]
    assert calls == ["king"]  # No new challenger transfer after cancellation.
    with cache_lock(tmp_path, cleanup=True) as free:
        assert free


def test_only_complete_snapshots_are_returned():
    with ModelDownloads(SimpleNamespace(resolve=lambda role: role + "-verified"), "king", "challenger") as downloads:
        assert downloads.result() == {"king": "king-verified", "challenger": "challenger-verified"}
        assert all(row["state"] == "verified" for row in downloads.status().values())
        assert downloads.elapsed_seconds >= 0
