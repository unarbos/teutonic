"""Small dependency-free helpers for validator startup decisions."""


def should_seed_king(force_seed_king: bool, state_king: dict | None) -> bool:
    return force_seed_king or not state_king
