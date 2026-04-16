#!/usr/bin/env python3
"""Check on-chain state for Teutonic (SN3)."""
import json
import sys

try:
    import bittensor as bt

    sub = bt.subtensor(network="finney")
    meta = sub.metagraph(3)
    block = sub.block

    wallet_name = "teutonic"
    hotkeys_info = []
    for hk_name in ["default", "h0", "h1", "h2", "h3", "h4", "h5"]:
        try:
            w = bt.wallet(name=wallet_name, hotkey=hk_name)
            uid = sub.get_uid_for_hotkey_on_subnet(w.hotkey.ss58_address, 3)
            hotkeys_info.append({
                "hotkey": hk_name,
                "uid": uid,
                "address": w.hotkey.ss58_address[:16] + "...",
                "registered": True,
            })
        except Exception:
            hotkeys_info.append({
                "hotkey": hk_name,
                "registered": False,
            })

    print(json.dumps({
        "block": block,
        "netuid": 3,
        "neurons": len(meta.hotkeys),
        "hotkeys": hotkeys_info,
    }, indent=2))
except Exception as e:
    print(json.dumps({"error": str(e)}))
