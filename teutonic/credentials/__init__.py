"""Credential lifecycle contracts."""

from .contracts import (
    ActivationSignal,
    activation_message,
    activation_signal_payload,
    latest_mailbox_object_key,
    mailbox_object_key,
    registration_id,
)

__all__ = [
    "ActivationSignal",
    "activation_message",
    "activation_signal_payload",
    "latest_mailbox_object_key",
    "mailbox_object_key",
    "registration_id",
]
