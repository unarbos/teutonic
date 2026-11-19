# RefineCode competition rollout — prepared, not activated

MAIN2 replaces MAIN. MIX01/MIX06 retire; MIX07 is unchanged. Historical rows,
reigns, manifests and source objects are retained.

| Competition | Mixture | Starting opponent |
| --- | --- | --- |
| MAIN2 | Original MAIN categories 93.325738958%; RefineCode 6.674261042% | Current MAIN king, pinned once at cutover |
| MIX07 | 100% DCLM | Existing king and configuration |
| MIX08 | 50% DCLM, 50% RefineCode | `teutonic-II-110B-A7B-5Eez-v9` |
| MIX09 | 100% RefineCode | `teutonic-II-110B-A7B-5Eez-v9` |

MAIN2 uses the existing MAIN registry counts plus 1,032,944,291,389 RefineCode
tokens: combined inventory 15,476,534,179,389. Each category weight is its count
divided by that total. These are sampling budget proportions, not score weights.
The original registry counts are preserved rather than silently recalculated.
All four competitions retain 60M MiMo input tokens, the existing 40/60 length
buckets, Horizon scoring and the 0.0005 bits/byte threshold.

## Data preparation (local files and read-only R2)

`collection.json` pins both audited dataset roots. A collection namespaces index
paths by source and leaves native shard IDs, UTF-8 byte offsets, token counts and
document IDs intact. Text is fetched from `dataset/teutonic-text/` or
`code/refinecode-text-v2/` with the existing ETag/range/SHA checks. No text copy or
retokenization is required. Old single-dataset requests remain supported.

Public RefineCode inventory: [manifest.json](https://pub-64e3f687d8a2411ca8595bc760d31654.r2.dev/refinecode-text-v2/manifest.json).
Resolve its object paths relative to `https://pub-64e3f687d8a2411ca8595bc760d31654.r2.dev/refinecode-text-v2/`.
This public download entry point does not replace the immutable source pins in
`collection.json`; evaluator credentials need read access to both source buckets.

Reuse the original sampling catalog only after verifying its existing pin:
`dataset/teutonic-text/sampling/0530605e7d6e54d2d00a7d6713f36cf959e42fbd78df32ee1d8aea1f9be60ed4/manifest.json`.
Its SHA-256 is `0530605e7d6e54d2d00a7d6713f36cf959e42fbd78df32ee1d8aea1f9be60ed4`.
Then run, with existing read credentials supplied through the environment:

```bash
.venv/bin/python scripts/prepare_packed_evaluation.py \
  --collection deployment/refinecode-20261119/collection.json \
  --source-catalog original=/path/to/verified-original-sampling.json \
  --competition main2 --competition mix08 --competition mix09 \
  --output-dir artifacts/refinecode-rollout --workers 64

.venv/bin/python scripts/publish_packed_evaluation.py \
  --bundle-dir artifacts/refinecode-rollout --dry-run
```

This builds the missing RefineCode catalog with resumable checkpoints and combines
the two catalogs into one immutable sampling artifact. Preparation writes only
local files. `--source-catalog refinecode=/path/to/catalog.json` can reuse a completed
RefineCode catalog. Do not run the single-source parallel preparation shortcut
for this collection. No completed catalog or publication is claimed by this folder.

## Activation, only after approval

1. Review the generated bundle, then publish it using the publisher above without
   `--dry-run` (publishing is its existing default). Verify all published hashes.
   Published artifacts go to `dataset/teutonic-text/`; the source datasets stay put.
2. Back up PostgreSQL using the existing control-plane backup procedure. Drain
   evaluations, submissions, promotions and outstanding weight attempts; stop
   the access controller, validator, promotion worker and weight publisher.
3. Install matching reader, validator, miner and website code. Apply
   `scripts/db/allow_main2_competition.sql` to allow the new identifier. This DDL
   alone does not activate anything.
4. Preview the cutover with the command below. It resolves v9 from stored king
   history, requiring one digest/owner and a finalized UID mapping. MAIN2 uses the
   current MAIN reign; MIX08/MIX09 share the resolved v9 starting reign. No new
   crown is fabricated and no old reign is ended by setup.
5. Review the exact starting pins and fixed reward allocations.
   Run the same command with a **new output path** and `--apply` only after approval.
   All competition, config and reward changes commit together. A repeat cutover
   refuses to overwrite existing destination competitions.
6. Refresh the finalized metagraph mapping and restart services in the normal
   deployment order. Verify MAIN2/MIX07/MIX08/MIX09 admission, pinned baselines,
   published dataset tables, rewards and the first full evaluation.

```bash
.venv/bin/python scripts/activate_refinecode_competitions.py \
  --bundle-dir artifacts/refinecode-rollout \
  --output artifacts/refinecode-cutover-preview.json
```

The versioned `refinecode-replacement-v6` reward policy keeps MAIN2 at 40%
and MIX07 at 20%. Retired MIX01/MIX06 kings retain 20% each until the first
winning challenger in MIX08 or MIX09 is crowned. Each first crown takes the
oldest remaining retired king's 20%, ordered by crown block and then reign ID.
Later wins replace only that competition's own recipient. The v9 starting
opponents do not receive incentives just for being pinned. Both new competitions
have 20% after their first wins. There is no latest-crown bonus, and positions
held by the same hotkey add together. Historical reward policy versions keep
their original meanings.

If the immediate v5 cutover was already applied, use the guarded
`scripts/activate_refinecode_replacement.py --output /path/to/preview.json`
repair with services stopped, review it, then use `--apply` with a new output
path. It requires no first crown in MIX08/MIX09 and performs no chain writes.

The historical MAIN database row remains the parent for service locks, reward
publications and chain scope, but `active=false` disables MAIN submissions.
MAIN2 is a separate child competition with its own history and starting pin.
Keep the existing service scope/environment name; do not rename it to MAIN2.
The validator permits an inactive parent only under the new reward policy.

The cutover command defaults to a read-only database transaction. `--apply`
requires stopped/drained services, a new before-state output file and hash-verified
published manifests. It performs no R2 uploads or chain submissions.

## Local validation

- Unit suite: 625 passed, 1 skipped; 17 subtests passed.
- Disposable PostgreSQL: scheduler and historical reward cutover regressions passed,
  including MAIN2/MIX08/MIX09 admission/crowning with a retired MAIN parent.
  The new configuration cutover also passed atomic commit and rollback checks.
- Website contracts: dashboard rendering, live data, competition overlaps and
  benchmark identities passed with the new competitions and RefineCode inventory.
- The updated reader loaded the hash-verified cached RefineCode root: 23,986 text
  shards, 23,986 indexes. Synthetic two-bucket sampling matched across worker counts
  and correctly routed colliding shard IDs and paths.

The wider suite exposed a cleanup race in failed document scoring. Reference and
paired preparation generators now close explicitly before their enclosing worker
contexts exit; failure tests check this without relying on garbage collection.
Scoring and sample membership are unchanged by that cleanup fix.

No production processes, R2 objects or production database rows were changed.
Real catalog preparation, model-pin resolution against production history, and a
GPU acceptance run on the new mixed dataset remain deployment prerequisites.

## Website history tabs

The main website now has adjacent loss-history and MAIN/Horizon comparison tabs.
The original loss chart remains the default and keeps its live feed and smoothing
controls. The comparison uses the approved 14-point, unfiltered historical snapshot
in `website/horizon-history.json`: the 13 backfilled MAIN kings plus Jasper's saved
production result. Both losses at each point use the same documents, weighted by
UTF-8 bytes. Samples differ between points; this is not a fixed benchmark.

The Horizon graph retains its legend and point tooltips, without the removed
subtitle and footer detail rows. Historical MAIN points remain MAIN points after
the MAIN2 rollout. New MAIN2 kings are appended automatically from the full published MAIN2 dashboard
history on the existing 15-second dashboard polling cycle, after publication.
Only public crowned winners with complete matched bits/byte losses are plotted.
The new king uses `avg_challenger_loss` and its reference uses `avg_reference_loss`;
both averages include all selected documents, irrespective of the reward filter.
Rejected/pending promotions and missing reference values are not plotted. Points
are deduplicated on refresh and reconstructed from the full history on page load.
No new inference, schema migration, or browser-local persistence is needed.
The horizontal axis keeps promotion order without numbered labels.
Deploy `horizon-history-data.js`, `horizon-history.js`, `horizon-history.css`, and
`horizon-history.json` with the rest of the website assets.
