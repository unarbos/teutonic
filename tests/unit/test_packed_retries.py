import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from urllib3.exceptions import ReadTimeoutError

from teutonic.evaluation import packed_text


def test_streaming_body_timeout_retries_pinned_range(monkeypatch):
    calls, bodies = [], []
    class TimeoutBody(io.BytesIO):
        def read(self, *args):
            raise ReadTimeoutError(None, None, 'fixture body timeout')
    class Client:
        def get_object(self, **kwargs):
            calls.append(kwargs)
            raw = TimeoutBody() if len(calls) == 1 else io.BytesIO(b'abcd')
            bodies.append(raw)
            return {'Body': StreamingBody(raw, 4), 'ETag': '"pin"',
                    'ContentRange': 'bytes 0-3/4'}
    monkeypatch.setattr(packed_text.time, 'sleep', lambda _: None)
    reader = packed_text.DatasetReader(Client())
    assert reader.read({'path': 'text/a.utf8', 'etag': '"pin"', 'size_bytes': 4}, 0, 4) == b'abcd'
    assert len(calls) == 2 and calls[0] == calls[1]
    assert all(body.closed for body in bodies) and not reader._objects


def test_r2_throttle_retries_exact_pinned_range_and_closes_body(monkeypatch):
    calls, sleeps, bodies = [], [], []
    obj = {"path": "text/a.utf8", "etag": '"pin"', "size_bytes": 100}
    class Client:
        def get_object(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= 2:
                raise ClientError({"Error": {"Code": "ServiceUnavailable"},
                                   "ResponseMetadata": {"HTTPStatusCode": 503}}, "GetObject")
            body = io.BytesIO(b"abcd")
            bodies.append(body)
            return {"Body": body, "ETag": obj["etag"], "ContentRange": "bytes 5-8/100"}
    monkeypatch.setattr(packed_text.time, "sleep", sleeps.append)
    reader = packed_text.DatasetReader(Client())
    assert reader.read(obj, 5, 4) == b"abcd"
    assert calls == [{"Bucket": "dataset", "Key": "teutonic-text/text/a.utf8",
                      "Range": "bytes=5-8", "IfMatch": '"pin"'}] * 3
    assert sleeps == [1, 2] and bodies[0].closed
    assert not reader._objects


@pytest.mark.parametrize("status,code,expected", [(403, "AccessDenied", 1),
                                                  (412, "PreconditionFailed", 1),
                                                  (503, "SlowDown", 8)])
def test_auth_etag_failures_are_not_retried_and_throttle_is_bounded(monkeypatch, status, code, expected):
    calls = []
    class Client:
        def get_object(self, **kwargs):
            calls.append(kwargs)
            raise ClientError({"Error": {"Code": code},
                               "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject")
    monkeypatch.setattr(packed_text.time, "sleep", lambda _: None)
    reader = packed_text.DatasetReader(Client())
    with pytest.raises(ClientError):
        reader.read({"path": "text/a.utf8", "etag": '"pin"', "size_bytes": 100}, 0, 4)
    assert len(calls) == expected and not reader._objects


def test_same_object_is_serialized_through_body_read_but_other_keys_overlap():
    entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    guard, active, maximum = threading.Lock(), 0, 0
    class Body(io.BytesIO):
        def read(self, *args):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
                first = not entered.is_set()
            if first:
                entered.set()
                assert release.wait(5)
            else:
                second_entered.set()
            value = super().read(*args)
            with guard:
                active -= 1
            return value
    class Client:
        def get_object(self, **kwargs):
            return {"Body": Body(b"abcd"), "ETag": '"pin"', "ContentRange": "bytes 0-3/4"}
    reader = packed_text.DatasetReader(Client())
    obj = {"path": "text/a.utf8", "etag": '"pin"', "size_bytes": 4}
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(reader.read, obj, 0, 4)
        assert entered.wait(5)
        second = pool.submit(reader.read, obj, 0, 4)
        assert not second_entered.wait(.05)
        other = pool.submit(reader.read, {**obj, "path": "text/b.utf8"}, 0, 4)
        try:
            assert second_entered.wait(5)
            assert other.result(timeout=5) == b"abcd"
        finally:
            release.set()
        assert first.result() == second.result() == b"abcd"
    assert maximum == 2 and not reader._objects
