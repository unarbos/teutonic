# Prepared 60M evaluation budget update

Status: local preparation only; no R2 publication or database changes.

MAIN, MIX01, MIX06 and MIX07 change from 35,000,000 to 60,000,000 original
MiMo input tokens. The 40/60 length quotas target 24M short and 36M long tokens.
Whole documents can overshoot quotas. The budget is measured before the
`horizon-deficit-active-bpb` scoring filter; skipped documents are not replaced.
The threshold stays at 0.00075 bits/byte.

Each new mix differs from the live configuration read on 13 November 2026 only
in `token_budget`. Dataset/catalog pins, category weights, sources and incentives
are preserved. The original `deployment/packed-text-20261112` release is historical
and must not be reapplied for this update.

`pending-publication.json` lists the four new immutable split objects and the
existing sampling catalog. It is a publication plan, not a receipt. The catalog
is deliberately not checked into Git. The complete local bundle, including the
catalog verified against the live SHA-256, is in
`artifacts/packed-text-60m-20261113/`.

For deployment from another checkout, copy this directory to a local artifact
folder and download the exact sampling object listed in `pending-publication.json`
as `sampling.json` there. The publisher validates its hash and all dependency pins.

After deployment approval, validate/publish that complete bundle using
`scripts/publish_packed_evaluation.py` (`--dry-run` validates without uploading;
publication is enabled by default). Save the resulting publication receipt and
verify every object before applying `cutover.sql`.

Drain evaluations and pending retries, stop claimers, and take/verify a database
backup first. Apply the SQL with `psql -v ON_ERROR_STOP=1` against the reviewed
netuid/generation. It checks the exact previous configuration versions in
`previous-configs.json`, the 35M budget, threshold, category weights and dataset
pins. It rejects drift atomically, retains the old configurations, preserves
early-stopping settings, and is idempotent. This updates configuration rows;
there is no SQL schema migration for this update.

Deploy matching evaluator/validator code with
`TEUTONIC_EVALUATION_POLICY_VERSION=horizon-deficit-active-bpb` on both hosts.
Confirm all four active budgets are 60M before resuming claims. For rollback,
restore the recorded prior active configuration IDs and matching code/policy
with services stopped; do not delete any manifests or historical rows.
