#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import chain_config

from miner import check_hotkey, commit_ready, get_upload_auth, register, upload_model
from miner.common import (
    AUTH_FILE,
    MANIFEST_FILE,
    REGISTRATION_FILE,
    RegistrationState,
    read_json,
    require_current_registration,
    subtensor_connection,
    wallet_from_args,
    write_json,
)
from teutonic.access.contracts import ReadySignal
from teutonic.config import DEFAULT_MAILBOX_PUBLIC_BASE_URL


SETTINGS_FILE = "settings.json"
SETTINGS_VERSION = 1


@dataclass(frozen=True, slots=True)
class SavedMiner:
    state_dir: Path
    registration: RegistrationState


def default_state_root() -> Path:
    configured = os.environ.get("TEUTONIC_MINER_STATE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    local = (Path.cwd() / ".teutonic-miner").resolve()
    source_tree = (Path(__file__).resolve().parents[1] / ".teutonic-miner").resolve()
    return local if local.exists() or not source_tree.exists() else source_tree


def load_settings(root: Path) -> dict[str, Any]:
    path = root / SETTINGS_FILE
    if not path.exists():
        return {"version": SETTINGS_VERSION}
    value = read_json(path)
    allowed = {"version", "active_hotkey", "wallet_path", "mailbox_base_url"}
    if set(value) - allowed or value.get("version") != SETTINGS_VERSION:
        raise RuntimeError(f"unsupported miner CLI settings: {path}")
    for field in allowed - {"version"}:
        if field in value and (not isinstance(value[field], str) or not value[field].strip()):
            raise RuntimeError(f"miner CLI setting {field} must be a non-empty string")
    return value


def save_settings(root: Path, value: Mapping[str, Any]) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    write_json(root / SETTINGS_FILE, value, secret=True)


def update_settings(root: Path, **updates: str | None) -> dict[str, Any]:
    settings = load_settings(root)
    for field, value in updates.items():
        if value is not None:
            settings[field] = str(value)
    save_settings(root, settings)
    return settings


def saved_miners(root: Path) -> list[SavedMiner]:
    if not root.exists():
        return []
    miners: list[SavedMiner] = []
    for registration_path in sorted(root.glob(f"*/{REGISTRATION_FILE}")):
        registration = RegistrationState.from_mapping(read_json(registration_path))
        miners.append(SavedMiner(registration_path.parent.resolve(), registration))
    return miners


def select_saved_miner(root: Path, selector: str | None = None) -> SavedMiner:
    miners = saved_miners(root)
    if not miners:
        raise RuntimeError(f"no saved miner registrations under {root}")
    selected = selector or load_settings(root).get("active_hotkey")
    if selected:
        matches = [
            miner
            for miner in miners
            if selected in {miner.registration.hotkey, miner.registration.hotkey_name}
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(f"no saved miner matches {selected!r}")
        raise RuntimeError(f"multiple saved miners match {selected!r}; use the hotkey address")
    if len(miners) == 1:
        return miners[0]
    names = ", ".join(miner.registration.hotkey_name for miner in miners)
    raise RuntimeError(f"multiple saved miners ({names}); run `teutonic-miner use HOTKEY`")


def wallet_arguments(miner: SavedMiner, wallet_path: Path) -> list[str]:
    state = miner.registration
    return [
        "--wallet-name",
        state.wallet_name,
        "--hotkey-name",
        state.hotkey_name,
        "--wallet-path",
        str(wallet_path),
        "--state-dir",
        str(miner.state_dir),
    ]


def eligibility_from_commitment(state: RegistrationState, commitment: str) -> str:
    if not commitment.startswith("r2ready:v1"):
        return "available"
    ready = ReadySignal.parse(
        commitment,
        signalling_hotkey=state.hotkey,
        block_number=0,
        extrinsic_index=0,
        event_index=0,
    )
    if ready.registration_id != state.registration_id:
        raise RuntimeError("finalized ready commitment belongs to another registration")
    return "consumed"


def finalized_eligibility(miner: SavedMiner, wallet_path: Path) -> str:
    state = miner.registration
    wallet_namespace = argparse.Namespace(
        wallet_name=state.wallet_name,
        hotkey_name=state.hotkey_name,
        wallet_path=wallet_path,
    )
    wallet = wallet_from_args(wallet_namespace)
    with subtensor_connection(state.network) as subtensor:
        current = require_current_registration(subtensor, saved=state, wallet=wallet)
        commitment = str(subtensor.get_commitment(state.netuid, current.uid) or "")
    return eligibility_from_commitment(current, commitment)


def require_available_eligibility(miner: SavedMiner, wallet_path: Path) -> None:
    if finalized_eligibility(miner, wallet_path) == "consumed":
        remove_local_upload_auth(miner)
        raise RuntimeError(
            "hotkey eligibility is permanently consumed; its mailbox credential is revoked "
            "and removed by the access controller"
        )


def remove_local_upload_auth(miner: SavedMiner) -> bool:
    try:
        (miner.state_dir / AUTH_FILE).unlink()
    except FileNotFoundError:
        return False
    return True


def configured_wallet_path(args: argparse.Namespace, settings: Mapping[str, Any]) -> Path:
    configured = args.wallet_path or settings.get("wallet_path") or os.environ.get("BT_WALLET_PATH")
    return Path(configured or "~/.bittensor/wallets").expanduser().resolve()


def print_saved_miner(miner: SavedMiner, *, active: bool) -> None:
    state = miner.registration
    marker = "*" if active else " "
    print(
        f"{marker} {state.hotkey_name} hotkey={state.hotkey} uid={state.uid} "
        f"network={state.network} netuid={state.netuid}"
    )


def add_selection_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hotkey", help="saved hotkey name or SS58 address")


def default_chain_generation() -> str:
    return os.environ.get("TEUTONIC_CHAIN_GENERATION", "").strip() or (
        chain_config.CHAIN_GENERATION
    )


def resolve_mailbox_base_url(
    explicit: str | None, settings: Mapping[str, Any]
) -> str:
    return (
        explicit
        or settings.get("mailbox_base_url")
        or os.environ.get("TEUTONIC_MAILBOX_PUBLIC_BASE_URL")
        or DEFAULT_MAILBOX_PUBLIC_BASE_URL
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teutonic-miner",
        description="Manage a Teutonic miner submission from saved local state.",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=default_state_root(),
        help="state root (default: .teutonic-miner or TEUTONIC_MINER_STATE_ROOT)",
    )
    parser.add_argument("--wallet-path", type=Path, help="Bittensor wallet root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list saved hotkeys")

    use = subparsers.add_parser("use", help="remember the active saved hotkey")
    use.add_argument("hotkey", help="saved hotkey name or SS58 address")

    configure = subparsers.add_parser("configure", help="save common non-secret settings")
    configure.add_argument("--mailbox-base-url")
    configure.add_argument("--wallet-path", type=Path, dest="configured_wallet_path")

    status = subparsers.add_parser("status", help="show local state without printing secrets")
    add_selection_argument(status)
    status.add_argument(
        "--local",
        action="store_true",
        help="skip the finalized-chain eligibility check",
    )

    check = subparsers.add_parser("check", help="validate the selected Ed25519 hotkey")
    add_selection_argument(check)
    check.add_argument("--wallet-name")
    check.add_argument("--hotkey-name")

    registration = subparsers.add_parser(
        "register", help="register if needed and commit mailbox activation"
    )
    add_selection_argument(registration)
    registration.add_argument("--wallet-name")
    registration.add_argument("--hotkey-name")
    registration.add_argument("--network")
    registration.add_argument("--netuid", type=int)
    registration.add_argument(
        "--chain-generation",
        default=default_chain_generation(),
        help="defaults to TEUTONIC_CHAIN_GENERATION or the active chain.toml",
    )
    registration.add_argument("--registration-timeout", type=int, default=600)
    registration.add_argument("--registration-tolerance", default="0.50")
    registration.add_argument("--check-only", action="store_true")

    auth = subparsers.add_parser("auth", help="retrieve and decrypt upload authorization")
    add_selection_argument(auth)
    auth.add_argument("--mailbox-base-url")
    auth.add_argument("--generation", type=int, help="pin a generation; default: latest")
    auth.add_argument("--timeout", type=int, default=600)

    upload = subparsers.add_parser("upload", help="refresh authorization and upload a model")
    add_selection_argument(upload)
    upload.add_argument("model_dir", type=Path)
    upload.add_argument("--name", required=True, dest="model_name")

    ready = subparsers.add_parser("ready", help="commit the uploaded model as ready")
    add_selection_argument(ready)
    ready.add_argument("--competition", choices=("main", "math", "code", "text"), default="main")

    submit = subparsers.add_parser(
        "submit", help="run check, register, auth, upload, and ready for a saved hotkey"
    )
    add_selection_argument(submit)
    submit.add_argument("--competition", choices=("main", "math", "code", "text"), default="main")
    submit.add_argument("model_dir", type=Path)
    submit.add_argument("--name", required=True, dest="model_name")
    submit.add_argument("--mailbox-base-url")
    submit.add_argument("--generation", type=int, help="pin a generation; default: latest")
    submit.add_argument("--auth-timeout", type=int, default=600)
    submit.add_argument("--registration-timeout", type=int, default=600)
    return parser


def run_register(
    args: argparse.Namespace,
    root: Path,
    settings: Mapping[str, Any],
    wallet_path: Path,
) -> SavedMiner:
    if bool(args.wallet_name) != bool(args.hotkey_name):
        raise RuntimeError("--wallet-name and --hotkey-name must be supplied together")
    if args.wallet_name:
        missing = [
            name
            for name in ("network", "netuid", "chain_generation")
            if getattr(args, name) in (None, "")
        ]
        if missing:
            raise RuntimeError("new registration requires --" + ", --".join(missing))
        wallet_namespace = argparse.Namespace(
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=wallet_path,
        )
        wallet = wallet_from_args(wallet_namespace)
        state_dir = (root / wallet.hotkey.ss58_address).resolve()
        command = [
            "--wallet-name",
            args.wallet_name,
            "--hotkey-name",
            args.hotkey_name,
            "--wallet-path",
            str(wallet_path),
            "--state-dir",
            str(state_dir),
            "--network",
            args.network,
            "--netuid",
            str(args.netuid),
            "--chain-generation",
            args.chain_generation,
        ]
        selector = wallet.hotkey.ss58_address
    else:
        saved = select_saved_miner(root, args.hotkey)
        state = saved.registration
        command = wallet_arguments(saved, wallet_path) + [
            "--network",
            state.network,
            "--netuid",
            str(state.netuid),
            "--chain-generation",
            state.chain_generation,
        ]
        selector = state.hotkey
    command += [
        "--registration-timeout",
        str(args.registration_timeout),
        "--registration-tolerance",
        args.registration_tolerance,
    ]
    if args.check_only:
        command.append("--check-only")
    register.main(command)
    saved = select_saved_miner(root, selector)
    update_settings(root, active_hotkey=saved.registration.hotkey, wallet_path=str(wallet_path))
    return saved


def run_auth(
    args: argparse.Namespace,
    miner: SavedMiner,
    root: Path,
    settings: Mapping[str, Any],
    wallet_path: Path,
) -> int:
    require_available_eligibility(miner, wallet_path)
    next_chain_check = 0.0

    def stop_if_revoked() -> None:
        nonlocal next_chain_check
        now = time.monotonic()
        if now < next_chain_check:
            return
        next_chain_check = now + 10.0
        require_available_eligibility(miner, wallet_path)

    mailbox_url = resolve_mailbox_base_url(args.mailbox_base_url, settings)
    update_settings(root, mailbox_base_url=mailbox_url, wallet_path=str(wallet_path))
    return get_upload_auth.main(
        wallet_arguments(miner, wallet_path)
        + [
            "--mailbox-base-url",
            mailbox_url,
            "--timeout",
            str(args.timeout),
        ] + (["--generation", str(args.generation)] if args.generation is not None else []),
        on_mailbox_not_found=stop_if_revoked,
    )


def dispatch(args: argparse.Namespace) -> int:
    root = args.state_root.expanduser().resolve()
    settings = load_settings(root)
    wallet_path = configured_wallet_path(args, settings)

    if args.command == "list":
        active = settings.get("active_hotkey")
        miners = saved_miners(root)
        if not miners:
            print(f"No saved miners under {root}")
            return 0
        for miner in miners:
            print_saved_miner(miner, active=miner.registration.hotkey == active)
        return 0

    if args.command == "use":
        miner = select_saved_miner(root, args.hotkey)
        update_settings(root, active_hotkey=miner.registration.hotkey)
        print_saved_miner(miner, active=True)
        return 0

    if args.command == "configure":
        configured_path = args.configured_wallet_path or args.wallet_path
        if not args.mailbox_base_url and not configured_path:
            raise RuntimeError("configure requires a wallet path or mailbox URL")
        settings = update_settings(
            root,
            mailbox_base_url=args.mailbox_base_url,
            wallet_path=str(configured_path.expanduser().resolve()) if configured_path else None,
        )
        print(f"settings_file={root / SETTINGS_FILE}")
        return 0

    if args.command == "register":
        run_register(args, root, settings, wallet_path)
        return 0

    if args.command == "check" and bool(args.wallet_name) != bool(args.hotkey_name):
        raise RuntimeError("--wallet-name and --hotkey-name must be supplied together")
    if args.command == "check" and args.wallet_name:
        return check_hotkey.main(
            [
                "--wallet-name",
                args.wallet_name,
                "--hotkey-name",
                args.hotkey_name,
                "--wallet-path",
                str(wallet_path),
            ]
        )

    miner = select_saved_miner(root, getattr(args, "hotkey", None))
    wallet_args = wallet_arguments(miner, wallet_path)

    if args.command == "status":
        print_saved_miner(miner, active=miner.registration.hotkey == settings.get("active_hotkey"))
        print(f"state_dir={miner.state_dir}")
        print(f"registration=saved")
        eligibility = "not_checked" if args.local else finalized_eligibility(miner, wallet_path)
        removed = remove_local_upload_auth(miner) if eligibility == "consumed" else False
        print(f"eligibility={eligibility}")
        if eligibility == "consumed":
            print("mailbox_credential=revoked_and_removed_by_controller")
        elif eligibility == "available":
            print("mailbox_credential=awaiting_or_available")
        else:
            print("mailbox_credential=not_checked")
        auth_state = "removed" if removed else (
            "present" if (miner.state_dir / AUTH_FILE).is_file() else "absent"
        )
        print(f"local_upload_auth={auth_state}")
        print(f"manifest={'present' if (miner.state_dir / MANIFEST_FILE).is_file() else 'absent'}")
        return 0
    if args.command == "check":
        return check_hotkey.main(wallet_args[:-2])
    if args.command == "auth":
        return run_auth(args, miner, root, settings, wallet_path)
    if args.command == "upload":
        run_auth(
            argparse.Namespace(mailbox_base_url=None, generation=None, timeout=600),
            miner, root, settings, wallet_path,
        )
        return upload_model.main(
            wallet_args
            + ["--model-dir", str(args.model_dir), "--model-name", args.model_name]
        )
    if args.command == "ready":
        return commit_ready.main(wallet_args + ["--competition", args.competition])
    if args.command == "submit":
        check_hotkey.main(wallet_args[:-2])
        register_args = argparse.Namespace(
            wallet_name=None,
            hotkey_name=None,
            hotkey=miner.registration.hotkey,
            registration_timeout=args.registration_timeout,
            registration_tolerance="0.50",
            check_only=False,
        )
        miner = run_register(register_args, root, settings, wallet_path)
        require_available_eligibility(miner, wallet_path)
        auth_args = argparse.Namespace(
            mailbox_base_url=args.mailbox_base_url,
            generation=args.generation,
            timeout=args.auth_timeout,
        )
        run_auth(auth_args, miner, root, settings, wallet_path)
        upload_model.main(
            wallet_arguments(miner, wallet_path)
            + ["--model-dir", str(args.model_dir), "--model-name", args.model_name]
        )
        return commit_ready.main(wallet_arguments(miner, wallet_path) + ["--competition", args.competition])
    raise RuntimeError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        return dispatch(build_parser().parse_args(argv))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
