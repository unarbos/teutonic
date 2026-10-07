from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from nacl.signing import SigningKey

from miner.cli import build_parser
from miner.get_upload_auth import ExpiredCredential, fetch_latest_credentials, validate_envelope
from teutonic.access.crypto import MailboxCipher, encode_ss58_public_key


def mailbox_fixture():
    key = SigningKey.generate()
    state = SimpleNamespace(
        netuid=3,
        uid=42,
        hotkey=encode_ss58_public_key(bytes(key.verify_key)),
        registration_id="b" * 64,
        chain_generation="test-chain",
        registration_block=100,
    )
    cipher = MailboxCipher(SigningKey.generate())

    def envelope(generation, *, expired=False, registration=None):
        return cipher.build_envelope(
            netuid=state.netuid,
            uid=state.uid,
            hotkey=state.hotkey,
            registration_id=registration or state.registration_id,
            generation=generation,
            endpoint="https://r2.example",
            private_model_bucket="private",
            allowed_prefix=f"models/registrations/{state.registration_id}/",
            access_key_id="access",
            secret_access_key="secret",
            session_token="session",
            expires_at=datetime.now(UTC) + timedelta(days=-1 if expired else 7),
            chain_generation=state.chain_generation,
            registration_block=state.registration_block,
        )

    def encrypted(value):
        return cipher.encrypt_for_hotkey(value, state.hotkey)

    return state, key, envelope, encrypted


def test_cli_defaults_to_latest_and_supports_explicit_generation():
    assert build_parser().parse_args(["auth"]).generation is None
    assert build_parser().parse_args(["auth", "--generation", "2"]).generation == 2
    assert build_parser().parse_args(["submit", "/tmp/model", "--name", "test"]).generation is None


def test_latest_encrypted_generation_is_discovered_and_verified():
    state, key, make, encrypt = mailbox_fixture()
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, content=encrypt(make(4)))

    client = httpx.Client(transport=httpx.MockTransport(handle))
    with patch("miner.get_upload_auth.httpx.Client", return_value=client):
        result = fetch_latest_credentials("https://mailbox.example", state, key, timeout=10)
    assert result["credential_generation"] == 4
    assert calls[0].url.path.endswith("/latest.bin")
    assert calls[0].url.params["poll"]


def test_legacy_generation_one_fallback():
    state, key, make, encrypt = mailbox_fixture()

    def handle(request):
        if request.url.path.endswith("/latest.bin"):
            return httpx.Response(404)
        return httpx.Response(200, content=encrypt(make(1)))

    client = httpx.Client(transport=httpx.MockTransport(handle))
    with patch("miner.get_upload_auth.httpx.Client", return_value=client):
        assert (
            fetch_latest_credentials("https://mailbox.example", state, key, timeout=10)[
                "credential_generation"
            ]
            == 1
        )


def test_expired_latest_waits_for_renewal_and_checks_eligibility():
    state, key, make, encrypt = mailbox_fixture()
    values = iter([make(1, expired=True), make(2)])
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=encrypt(next(values)))
        )
    )
    waited = []
    with (
        patch("miner.get_upload_auth.httpx.Client", return_value=client),
        patch("miner.get_upload_auth.time.sleep"),
    ):
        result = fetch_latest_credentials(
            "https://mailbox.example",
            state,
            key,
            timeout=10,
            on_wait=lambda: waited.append(True),
        )
    assert result["credential_generation"] == 2
    assert waited == [True]


def test_latest_rejects_wrong_registration_without_fallback():
    state, key, make, encrypt = mailbox_fixture()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=encrypt(make(2, registration="a" * 64)))
        )
    )
    with (
        patch("miner.get_upload_auth.httpx.Client", return_value=client),
        pytest.raises(RuntimeError, match="registration_id"),
    ):
        fetch_latest_credentials("https://mailbox.example", state, key, timeout=10)


def test_explicit_generation_remains_pinned_and_expired_credentials_fail():
    state, _, make, _ = mailbox_fixture()
    with pytest.raises(RuntimeError, match="credential_generation"):
        validate_envelope(make(2), state, 1)
    with pytest.raises(ExpiredCredential):
        validate_envelope(make(1, expired=True), state, 1)
