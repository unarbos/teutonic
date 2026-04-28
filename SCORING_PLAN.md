# Exponential Dethrone Scoring — Implementation Plan (simplified)

## Overview

Replace `TOPK_WEIGHTS = [0.75, 0.19, 0.06]` with `exp(k · i)` over the **last N kings**, where `i` is each king's position in the sliding window (0 = oldest, N-1 = newest). State is a list of at most N entries kept inside `State`, persisted alongside the rest of `state/validator_state.json`. New section on the website shows the resulting weight distribution.

## Mechanism

Maintain in `State`:

```python
self.king_history: list[dict] = []
# each entry: {hotkey, hf_repo, king_revision, crowned_block, crowned_at, challenge_id}
# capped at MAX_KING_HISTORY, oldest at index 0, newest last
```

On every accepted dethrone, append the new king and trim to the last N. Compute weights as:

```
raw_i = exp(k · i)        for i = 0 .. n-1   (n = len(king_history), n ≤ N)
weight_i = raw_i / Σ raw_j
```

With defaults `k=1.0`, `N=16`: newest king ≈ 63%, 2nd ≈ 23%, 3rd ≈ 8.6%, 4th ≈ 3.2%, ..., 16th ≈ 5e-8 (negligible). All values fit comfortably in float64 — `exp(1.0 · 15) ≈ 3.3e6`.

Env vars:

- `TEUTONIC_KING_HISTORY` (default `16`) — window size N
- `TEUTONIC_SCORE_K` (default `1.0`) — exponential bias

## Files touched

- [teutonic/validator.py](teutonic/validator.py) — add `king_history` to `State`, append/trim on dethrone, derive weights inside `maybe_set_weights`, expose in dashboard payload.
- [teutonic/index.html](teutonic/index.html) — new "Weight Distribution" section.

## Step 1 — Add window + helpers to `State`

In `State.__init__`:

```python
self.king_history: list[dict] = []
```

In `State.load`, after the existing `validator_state.json` block:

```python
self.king_history = st.get("king_history", []) if st else []
if not self.king_history:
    self._backfill_king_history_from_jsonl()
```

In `State.flush`, add `"king_history": self.king_history` to the `validator_state.json` payload.

New methods on `State`:

```python
def append_king_history(self, hotkey, hf_repo, king_revision,
                        crowned_block, challenge_id):
    self.king_history.append({
        "hotkey": hotkey,
        "hf_repo": hf_repo,
        "king_revision": king_revision,
        "crowned_block": crowned_block,
        "challenge_id": challenge_id,
        "crowned_at": _now(),
    })
    if len(self.king_history) > MAX_KING_HISTORY:
        self.king_history = self.king_history[-MAX_KING_HISTORY:]

def king_history_weights(self) -> list[dict]:
    """exp(SCORE_K * i) softmax over king_history; returns oldest-first list
    of {hotkey, hf_repo, position, weight}."""
    n = len(self.king_history)
    if n == 0:
        return []
    # shift so newest = 0, oldest = -(n-1) for numerical stability
    raws = [math.exp(SCORE_K * (i - (n - 1))) for i in range(n)]
    s = sum(raws)
    out = []
    for i, (entry, raw) in enumerate(zip(self.king_history, raws)):
        out.append({
            "hotkey": entry["hotkey"],
            "hf_repo": entry.get("hf_repo", ""),
            "position": i,
            "weight": raw / s,
        })
    return out
```

## Step 2 — One-shot backfill from `state/history.jsonl`

So historical kings get weight on first boot, scan the existing event log once and pick up the last `MAX_KING_HISTORY` `king_changed` events.

```python
def _backfill_king_history_from_jsonl(self):
    log.info("king_history empty; backfilling from state/history.jsonl ...")
    try:
        raw = self.r2.client.get_object(
            Bucket=R2_BUCKET, Key="state/history.jsonl"
        )["Body"].read()
    except Exception:
        log.warning("no history.jsonl to backfill from; king_history starts empty")
        return
    events = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "king_changed":
            continue
        if ev.get("challenge_id") == "seed":
            continue
        events.append(ev)
    events = events[-MAX_KING_HISTORY:]
    dh = self.r2.get("state/dashboard_history.json") or {}
    by_cid = {h.get("challenge_id"): h
              for h in dh.get("history", []) if h.get("accepted")}
    for ev in events:
        h = by_cid.get(ev.get("challenge_id", ""), {})
        self.king_history.append({
            "hotkey": ev.get("hotkey", ""),
            "hf_repo": h.get("challenger_repo", ""),
            "king_revision": "",
            "crowned_block": 0,
            "challenge_id": ev.get("challenge_id", ""),
            "crowned_at": ev.get("timestamp", _now()),
        })
    # Make sure the current king (if any) is the last entry
    if self.king and self.king.get("hotkey"):
        if not self.king_history or self.king_history[-1]["hotkey"] != self.king["hotkey"]:
            self.append_king_history(
                self.king.get("hotkey", ""),
                self.king.get("hf_repo", ""),
                self.king.get("king_revision", ""),
                self.king.get("crowned_block", 0),
                self.king.get("challenge_id", "backfill-current"),
            )
    log.info("king_history backfilled with %d entries", len(self.king_history))
```

## Step 3 — Append on dethrone

In `process_challenge`, inside the `if became_new_top1:` branch, right after `state.set_king(...)`:

```python
state.append_king_history(hotkey, hf_repo, challenger_revision,
                          dethrone_block, cid)
```

## Step 4 — Replace `set_weights` source of truth

At the top of `validator.py` near other env constants:

```python
SCORE_K = float(os.environ.get("TEUTONIC_SCORE_K", "1.0"))
MAX_KING_HISTORY = int(os.environ.get("TEUTONIC_KING_HISTORY", "16"))
```

In `maybe_set_weights`, replace:

```python
ranked = state.topk_for_weight_set()
ranked_hotkeys = [e["hotkey"] for e in ranked]
ranked_weights = _normalize_weights(TOPK_WEIGHTS, len(ranked_hotkeys))
```

with:

```python
weight_entries = state.king_history_weights()
ranked_hotkeys = [w["hotkey"] for w in weight_entries]
ranked_weights = [w["weight"] for w in weight_entries]
```

`set_weights` already filters to hotkeys present in the metagraph and renormalizes, so deregistered past kings get dropped silently and their share spreads over the survivors. The fallback `if not ranked_hotkeys: ranked_hotkeys = [fallback_hotkey]; ranked_weights = [1.0]` stays as-is.

The `TOPK_WEIGHTS` constant and `topk_for_weight_set` are no longer referenced anywhere — leave the score-window machinery in place (it's used by the existing dashboard) but the weight path is fully ledger-driven now.

## Step 5 — Dashboard payload

In `flush_dashboard`, add to `payload`:

```python
payload["king_history"] = {
    "k": SCORE_K,
    "max": MAX_KING_HISTORY,
    "weights": list(reversed(state.king_history_weights())),  # newest first for UI
}
```

## Step 6 — Website: Weight Distribution section in `index.html`

Insert after the "Last Winner" section:

```html
<div>
    <div class="section-label">Weight Distribution (exp(k·i) over last N kings)</div>
    <div id="weight-meta" style="font-size:11px;opacity:0.7;margin-bottom:6px;">--</div>
    <div id="weight-bars" style="display:flex;flex-direction:column;gap:3px;font-size:11px;"></div>
</div>
```

In `render(d)`:

```javascript
var kh = d.king_history || {};
var ws = kh.weights || [];
document.getElementById("weight-meta").textContent =
    "k=" + (kh.k || "?") + "  N=" + (kh.max || "?") +
    "  active=" + ws.length;
var wbars = document.getElementById("weight-bars");
if (ws.length === 0) {
    wbars.innerHTML = '<div style="opacity:0.6">No dethrones recorded yet</div>';
} else {
    var maxW = ws[0].weight;
    wbars.innerHTML = ws.map(function(w, idx) {
        var pct = (w.weight * 100).toFixed(2);
        var barW = (w.weight / maxW * 100).toFixed(1);
        var repo = w.hf_repo || (w.hotkey ? short(w.hotkey, 12) : "--");
        var link = w.hf_repo
            ? '<a href="https://huggingface.co/' + w.hf_repo + '" target="_blank" style="color:var(--ink)">' + repo + '</a>'
            : repo;
        var label = idx === 0 ? "NOW" : "-" + idx;
        return '<div style="display:flex;align-items:center;gap:8px;">' +
            '<span style="width:42px;text-align:right;opacity:0.6">' + label + '</span>' +
            '<div style="flex:1;background:var(--bg-muted,#eee);height:14px;position:relative;">' +
              '<div style="background:var(--ink);height:100%;width:' + barW + '%;"></div>' +
            '</div>' +
            '<span style="width:64px;text-align:right;">' + pct + '%</span>' +
            '<span style="flex:2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + link + '</span>' +
        '</div>';
    }).join("");
}
```

The dashboard `index.html` is uploaded to Hippius by `main()` on every startup (validator.py lines 1509-1514), so the new HTML deploys automatically on `pm2 restart`.

## Step 7 — Restart

```bash
pm2 restart teutonic-validator
pm2 logs teutonic-validator --lines 200
```

Expected log lines:

- `king_history empty; backfilling from state/history.jsonl ...`
- `king_history backfilled with N entries`
- `setting weights at block ... (forced, startup) to <hk_newest>:0.63, <hk_n-1>:0.23, ...`
- `uploaded dashboard to Hippius`

Then check `https://s3.hippius.com/teutonic-sn3/index.html` for the new section.

## Numerical safety

With `N=16, k=1`: max raw value is `exp(0) = 1`, min is `exp(-15) ≈ 3e-7`. Sum ≈ 1.58. No overflow possible. Even at `k=10, N=16` (extreme): max = 1, min = `exp(-150) ≈ 7e-66` (underflows harmlessly to 0), sum ≈ 1. Safe at any reasonable parameter setting.

## Out of scope

- Eval method (paired bootstrap LCB) — untouched.
- `score_window` / `topk` machinery — left in place; harmless dead weight in `validator_state.json`.
- Penalized eval path — untouched.
- Tests / docs (per workspace rules).

## Risks

- If `history.jsonl` is missing/corrupt, backfill falls back to "current king only" and the next 15 dethrones refill the window.
- Past kings whose hotkeys deregistered get silently filtered by `set_weights`; their share redistributes.

## Todo checklist

- [ ] Add `SCORE_K`, `MAX_KING_HISTORY` env constants
- [ ] Add `king_history` list + load/flush + `append_king_history` + `king_history_weights` + `_backfill_king_history_from_jsonl` to `State`
- [ ] Append to `king_history` after `state.set_king` in `process_challenge`
- [ ] Swap `topk_for_weight_set` for `king_history_weights` in `maybe_set_weights`
- [ ] Add `king_history` block to dashboard payload
- [ ] Add Weight Distribution section + bar renderer to `teutonic/index.html`
- [ ] `pm2 restart teutonic-validator` and confirm logs + dashboard
