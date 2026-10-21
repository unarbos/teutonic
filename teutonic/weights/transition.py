"""Administrative snapshot of the live reward positions; no chain writes."""

from .policy import MAIN_SHARES, replacement_rewards


def snapshot_transition(connection, main):
    """Called under the MAIN row lock after all pending work has drained."""
    if main["reward_transition"] is not None:
        raise ValueError("reward transition is already initialized")
    from teutonic.access.repository import CONTROLLER_ADVISORY_LOCK_ID
    from teutonic.promotion.repository import promotion_worker_lock_key
    from teutonic.validator.repository import scheduler_lock_key
    from teutonic.weights.repository import weight_publisher_lock_key

    scope = (main["netuid"], main["chain_generation"], main["name"])
    locks = {
        "access controller": CONTROLLER_ADVISORY_LOCK_ID,
        "validator": scheduler_lock_key(*scope),
        "promotion worker": promotion_worker_lock_key(*scope),
        "weight publisher": weight_publisher_lock_key(*scope),
    }
    for service, lock in locks.items():
        if not connection.execute(
            "SELECT pg_try_advisory_xact_lock(%s) AS acquired", (lock,)
        ).fetchone()["acquired"]:
            raise ValueError(f"stop the {service} before cutover")
    pending = connection.execute(
        """SELECT count(*) AS n FROM control_plane.uploads u
           JOIN control_plane.registrations r USING(registration_id)
           WHERE r.netuid=%s AND r.chain_generation=%s AND u.state IN
           ('ready_signaled','verifying','immutable_snapshot_created','ready_for_evaluation',
            'retry_pending','evaluation_claimed','evaluating','accepted_pending_promotion','promoted')""",
        (main["netuid"], main["chain_generation"]),
    ).fetchone()["n"]
    if pending:
        raise ValueError("drain all submissions and promotions before cutover")
    kings = connection.execute(
        """SELECT c.competition_key, k.* FROM control_plane.competitions c
           JOIN control_plane.king_reigns k ON k.reign_id=c.current_reign_id
           WHERE c.competition_id=%s OR c.main_competition_id=%s""",
        (main["competition_id"], main["competition_id"]),
    ).fetchall()
    by_key = {r["competition_key"]: r for r in kings}
    if "main" not in by_key or set(by_key) - {"main", "math", "code", "text"}:
        raise ValueError("cutover needs a MAIN king and no existing MIX kings")
    publication = connection.execute(
        "SELECT * FROM control_plane.weight_publications WHERE source_reign_id=%s",
        (main["reward_reign_id"] or main["current_reign_id"],),
    ).fetchone()
    if not publication or not publication["policy_weights"]:
        raise ValueError("cutover needs an explicit durable reward policy")
    current = {}
    for h, w in zip(publication["policy_hotkeys"], publication["policy_weights"], strict=True):
        current[h] = current.get(h, 0) + float(w)
    total = sum(current.values())
    current = {h: w / total for h, w in current.items()}
    history = main["reward_main_hotkeys"] or publication["policy_hotkeys"]
    shares = MAIN_SHARES[len(by_key) - 1]
    history = list(dict.fromkeys(history))[: len(shares)]
    scale = sum(shares) / sum(shares[: len(history)])
    positions = [
        (f"main-history:{i}", h, share * scale / 100)
        for i, (h, share) in enumerate(zip(history, shares))
    ]
    positions += [(k, row["hotkey"], 0.15) for k, row in by_key.items() if k != "main"]
    combined = {}
    for _, h, w in positions:
        combined[h] = combined.get(h, 0) + w
    # Bootstrap recipients may have no reign yet: preserve actual durable shares.
    if set(combined) != set(current) or any(
        abs(combined.get(h, 0) - w) > 1e-9 for h, w in current.items()
    ):
        positions = [(f"snapshot:{i}", h, w) for i, (h, w) in enumerate(current.items())]
    main_hotkey = by_key["main"]["hotkey"]
    primary = next((i for i, (_, h, _) in enumerate(positions) if h == main_hotkey), None)
    if primary is None:
        raise ValueError("latest MAIN king is absent from the current reward policy")
    entries = []
    for i, (identity, hotkey, weight) in enumerate(positions):
        reign = connection.execute(
            """SELECT k.reign_id, k.crowned_finalized_block FROM control_plane.king_reigns k
               JOIN control_plane.competitions c USING(competition_id)
               WHERE (c.competition_id=%s OR c.main_competition_id=%s) AND k.hotkey=%s
                 AND (NOT %s OR c.competition_id=%s)
               ORDER BY k.crowned_finalized_block DESC, k.reign_id LIMIT 1""",
            (
                main["competition_id"],
                main["competition_id"],
                hotkey,
                identity.startswith("main-history:"),
                main["competition_id"],
            ),
        ).fetchone()
        if identity in by_key:
            reign = by_key[identity]
        entries.append(
            {
                "id": identity,
                "hotkey": hotkey,
                "weight": weight,
                "reign_id": str(reign["reign_id"]) if reign else None,
                "crowned_block": reign["crowned_finalized_block"] if reign else -1,
            }
        )
    result = {
        "version": "mix-replacement-v1",
        "main": entries[primary],
        "legacy": sorted(
            [e for i, e in enumerate(entries) if i != primary],
            key=lambda e: (e["crowned_block"], e["id"]),
        ),
        "mixes": {},
        "evictions": [],
    }
    hotkeys, weights = replacement_rewards(result)
    if any(abs(dict(zip(hotkeys, weights)).get(h, 0) - w) > 1e-9 for h, w in current.items()):
        raise ValueError("cutover would change the current reward distribution")
    return result
