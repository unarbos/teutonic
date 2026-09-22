"use strict";

const assert = require("assert");
const dashboard = require("../../website/dashboard-v1.js");

function base() {
    return {
        schema_version: 1,
        publication_id: "11111111-1111-4111-8111-111111111111",
        generated_at: "2026-08-18T12:00:00Z",
        updated_at: "2026-08-18T12:00:00Z",
        source_watermark: 1,
        chain: {
            name: "Teutonic", netuid: 306, generation: "test", competition: "quasar",
            seed_repo: "owner/genesis", seed_digest: "hf:" + "b".repeat(40), seed_repo_backend: "hf"
        },
        king: null,
        king_payout: { weight: null, alpha_per_hour: null, usd_per_hour: null },
        king_chain: [],
        stats: {},
        current_eval: null,
        queue: [],
        history: [],
        dataset_versions: [],
        weight_status: {},
        service_status: { overall: "healthy" },
        market: null
    };
}

function hiddenRecord() {
    return { model_identity: "hidden_until_promotion", challenger_repo: null };
}

const idle = dashboard.presentation(base());
assert.strictEqual(idle.idle, true);
assert.strictEqual(idle.historyEmpty, true);

const queuedPayload = base();
queuedPayload.queue = [hiddenRecord()];
const queued = dashboard.presentation(queuedPayload);
assert.strictEqual(queued.idle, false);
assert.strictEqual(queued.duelIdentity, dashboard.HIDDEN);

const evaluatingPayload = base();
evaluatingPayload.current_eval = hiddenRecord();
assert.strictEqual(dashboard.presentation(evaluatingPayload).duelIdentity, dashboard.HIDDEN);
assert.deepStrictEqual(
    dashboard.currentEvaluationPresentation({
        provisional_mu_hat: 0.72,
        provisional_lcb: 0.61,
        delta_threshold: 0.5,
        provisional_n_sequences: 400,
        provisional_n_bootstrap: 1000
    }),
    {
        available: true,
        lcb: 0.61,
        muHat: 0.72,
        threshold: 0.5,
        sequences: 400,
        bootstraps: 1000,
        clearsThreshold: true
    }
);
assert.strictEqual(dashboard.currentEvaluationPresentation({}).available, false);
assert.strictEqual(
    dashboard.currentEvaluationPresentation({ provisional_lcb: null }).available,
    false
);

const rejectedPayload = base();
rejectedPayload.history = [hiddenRecord()];
assert.strictEqual(dashboard.presentation(rejectedPayload).historyIdentity[0], dashboard.HIDDEN);

const winnerPayload = base();
winnerPayload.history = [{ model_identity: "public", challenger_repo: "owner/winner" }];
assert.strictEqual(dashboard.presentation(winnerPayload).historyIdentity[0], "owner/winner");

const nonWinnerPayload = base();
nonWinnerPayload.history = [{ model_identity: "public", challenger_repo: "owner/non-winner" }];
assert.strictEqual(dashboard.presentation(nonWinnerPayload).historyIdentity[0], "owner/non-winner");

const degradedPayload = base();
degradedPayload.service_status.overall = "degraded";
assert.strictEqual(dashboard.presentation(degradedPayload).degraded, true);

const staleMarketPayload = base();
staleMarketPayload.market = { stale: true };
assert.strictEqual(dashboard.presentation(staleMarketPayload).marketStale, true);

const historyRows = [
    { verdict: "accepted", challenge_id: "accepted" },
    { verdict: "error", challenge_id: "failed" },
    { verdict: "rejected", challenge_id: "rejected" }
];
const hiddenErrors = dashboard.historyPresentation(historyRows, false);
assert.deepStrictEqual(hiddenErrors.rows.map((row) => row.challenge_id), ["accepted", "rejected"]);
assert.strictEqual(hiddenErrors.errorCount, 1);
assert.strictEqual(dashboard.historyPresentation(historyRows, true).rows.length, 3);
assert.strictEqual(dashboard.verdictLabel("accepted"), "CHALLENGER");
assert.strictEqual(dashboard.verdictLabel("rejected"), "KING");
assert.strictEqual(dashboard.verdictLabel("error"), "ERROR");
assert.deepStrictEqual(
    dashboard.evaluationHistoryMetricsPresentation({
        n_sequences_evaluated: 600,
        n_sequences: 2000,
        early_stopped: true
    }),
    {
        samples: "600 / 2000",
        samplesTitle: "600 of 2000 samples evaluated",
        earlyStopped: true,
        earlyStopLabel: "YES"
    }
);
assert.strictEqual(
    dashboard.evaluationHistoryMetricsPresentation({ n_sequences: 2000 }).samples,
    "2000"
);
assert.deepStrictEqual(
    dashboard.evaluationHistoryMetricsPresentation({}),
    {
        samples: "--",
        samplesTitle: "Sample count unavailable",
        earlyStopped: false,
        earlyStopLabel: "--"
    }
);
assert.deepStrictEqual(
    dashboard.sourceScoresPresentation({
        source_scores: [
            { source: "finewebedu", n_sequences: 600, avg_king_loss: 2.1, avg_challenger_loss: 2.09, mu_hat: 0.01 },
            { source: "broken", n_sequences: null, avg_king_loss: 1, avg_challenger_loss: 1, mu_hat: 0 }
        ]
    }),
    {
        count: 1,
        rows: [{ source: "finewebedu", nSequences: 600, kingLoss: 2.1, challengerLoss: 2.09, muHat: 0.01 }]
    }
);

assert.strictEqual(
    dashboard.taoMarketCapHotkeyUrl("5MinerHotkey"),
    "https://taomarketcap.com/hotkey/5MinerHotkey/metagraph"
);
assert.strictEqual(
    dashboard.taoMarketCapHotkeyUrl("5Miner Hotkey"),
    "https://taomarketcap.com/hotkey/5Miner%20Hotkey/metagraph"
);
assert.strictEqual(dashboard.taoMarketCapHotkeyUrl(""), "");
assert.strictEqual(
    dashboard.taoMarketCapColdkeyUrl("5Ek5KoE56Y5vj4gDMLARUS6UmKhPZBJBS7z2aBkQWZtr57gG"),
    "https://taomarketcap.com/coldkey/5Ek5KoE56Y5vj4gDMLARUS6UmKhPZBJBS7z2aBkQWZtr57gG"
);
assert.strictEqual(
    dashboard.taoMarketCapColdkeyUrl("5Cold Key"),
    "https://taomarketcap.com/coldkey/5Cold%20Key"
);
assert.strictEqual(dashboard.taoMarketCapColdkeyUrl(""), "");

const shards = dashboard.shardPresentation({
    shards_used: [
        { source: "finewebedu", names: ["part-001.npy", "part-002.npy", "part-001.npy"] },
        { source: " ", names: ["part-003.npy", ""] },
        { source: "ignored", names: [] }
    ]
});
assert.strictEqual(shards.count, 3);
assert.deepStrictEqual(shards.groups, [
    { source: "finewebedu", names: ["part-001.npy", "part-002.npy"] },
    { source: "dataset", names: ["part-003.npy"] }
]);
assert.deepStrictEqual(dashboard.shardPresentation({}).groups, []);

assert.deepStrictEqual(
    dashboard.uploadFailurePresentation({
        uid: 170,
        registration_state: "active",
        upload_id: "9452d08b-b3bb-4bc9-9c45-01889fff6fa8",
        upload_state: "verification_failed",
        error_code: "ArtifactIntegrityError"
    }),
    {
        registration: "UID 170 · ACTIVE",
        uploadId: "9452d08b-b3bb-4bc9-9c45-01889fff6fa8",
        uploadState: "verification_failed",
        failureCode: "ArtifactIntegrityError"
    }
);
assert.strictEqual(dashboard.uploadFailurePresentation({ verdict: "error" }), null);

assert.deepStrictEqual(
    dashboard.decisionPresentation({ verdict: "accepted", lcb: 0.64, delta: 0.5 }),
    {
        kind: "win",
        label: "WIN REASON",
        summary: "LCB 0.640000 > REQUIRED 0.500000 · MARGIN +0.140000",
        detail: "The confidence-adjusted improvement was high enough to replace the king."
    }
);
assert.deepStrictEqual(
    dashboard.decisionPresentation({ verdict: "rejected", lcb: 0.38, delta_threshold: 0.5 }),
    {
        kind: "loss",
        label: "LOSS REASON",
        summary: "LCB 0.380000 ≤ REQUIRED 0.500000 · SHORTFALL 0.120000",
        detail: "The measured improvement was not confident enough to replace the king."
    }
);
assert.strictEqual(dashboard.decisionPresentation({ verdict: "error" }).kind, "error");

const invalid = base();
invalid.schema_version = 2;
assert.throws(() => dashboard.presentation(invalid), /unsupported dashboard schema/);

const datasetManifest = {
    schema_version: 1,
    dataset_label: "fixture-mix",
    eval_n: 300,
    sources: [{
        name: "fixture",
        proportion: 1,
        manifest_url: "https://datasets.example/fixture/manifest.json",
        manifest_sha256: "b".repeat(64),
        source_repo: "owner/dataset",
        tokenizer: "owner/tokenizer",
        dtype: "uint32",
        tokenization_mode: "seq_packed_shards",
        sequence_length: 2048,
        total_tokens: 2_048_000,
        total_shards: 4,
        estimated_sequences: 1000
    }]
};
const dataset = dashboard.datasetPresentation(datasetManifest);
assert.strictEqual(dataset.rows.length, 1);
assert.strictEqual(dataset.totalTokens, 2_048_000);
assert.strictEqual(dataset.totalSequences, 1000);
assert.strictEqual(dataset.evalTokens, 614_400);
assert.strictEqual(dataset.rows[0].normalizedWeight, 1);
assert.strictEqual(dataset.rows[0].source, "owner/dataset");
assert.strictEqual(dataset.rows[0].metadataLoaded, true);
assert.throws(() => dashboard.datasetPresentation({}, {}), /sources must be an array/);

const benchmarkResults = dashboard.benchmarkPresentation({
    schema_version: "teutonic-king-benchmark-all-results.v2",
    generated_at: "2026-08-26T10:30:39Z",
    benchmark_result_count: 3,
    kings: [
        {
            king_id: "reign-6",
            status: "partial",
            updated_at: "2026-08-26T09:00:00Z",
            result: {
                model: { reign_number: 6, uid: 22, hotkey: "prior", model_repo: "owner/prior" },
                benchmarks: [
                    { name: "BBH", fewshot: 3, status: "completed", metric: { name: "acc_norm,none", value: 0.25 } },
                    { name: "MMLU", fewshot: 0, status: "completed", metric: { name: "acc,none", value: 0.2 } }
                ]
            }
        },
        {
            king_id: "reign-7",
            status: "completed",
            updated_at: "2026-08-26T10:29:18Z",
            result: {
                model: { reign_number: 7, uid: 226, hotkey: "current", model_repo: "owner/current", is_current: true },
                benchmarks: [
                    { name: "BBH", fewshot: 3, status: "completed", metric: { name: "acc_norm,none", value: 0.297344 } },
                    { name: "MMLU_v2", fewshot: 5, status: "completed", metric: { name: "acc,none", value: 0.7395 } },
                    { name: "HellaSwag_v2", fewshot: 10, status: "completed", metric: { name: "acc_norm,none", value: 0.8419 } },
                    { name: "WinoGrande_v2", fewshot: 5, status: "completed", metric: { name: "acc,none", value: 0.7695 } },
                    { name: "GSM8K", fewshot: 4, status: "completed", metric: { name: "exact_match,strict-match", value: 0 } },
                    { name: "ARC-C_v2", fewshot: 25, status: "completed", metric: { name: "acc_norm,none", value: 0.657 } },
                    { name: "GPQA Diamond_v2", fewshot: 5, status: "completed", metric: { name: "acc,none", value: 0.3535 } },
                    { name: "MATH-500", fewshot: 4, status: "completed", metric: { name: "exact_match,none", value: 0.38 } }
                ]
            }
        }
    ]
});
assert.deepStrictEqual(benchmarkResults.kings.map((king) => king.kingId), ["reign-7", "reign-6"]);
assert.strictEqual(benchmarkResults.selected.kingId, "reign-7");
assert.strictEqual(benchmarkResults.selected.benchmarks.length, 10);
assert.strictEqual(benchmarkResults.selected.benchmarks[0].name, "BBH");
assert.strictEqual(benchmarkResults.selected.benchmarks[0].score, 0.297344);
assert.deepStrictEqual(
    benchmarkResults.selected.benchmarks.slice(1, 4).map((benchmark) => [benchmark.name, benchmark.score, benchmark.fewshot]),
    [["MMLU", 0.7395, 5], ["HellaSwag", 0.8419, 10], ["WinoGrande", 0.7695, 5]]
);
assert.strictEqual(benchmarkResults.selected.benchmarks[4].name, "GSM8K");
assert.strictEqual(benchmarkResults.selected.benchmarks[4].score, 0);
assert.deepStrictEqual(
    [benchmarkResults.selected.benchmarks[6].name, benchmarkResults.selected.benchmarks[6].score, benchmarkResults.selected.benchmarks[6].fewshot],
    ["ARC-C", 0.657, 25]
);
assert.strictEqual(benchmarkResults.selected.benchmarks[8].name, "GPQA Diamond");
assert.strictEqual(benchmarkResults.selected.benchmarks[8].score, 0.3535);
assert.strictEqual(benchmarkResults.selected.benchmarks[8].fewshot, 5);
assert.strictEqual(benchmarkResults.selected.benchmarks[9].name, "MATH-500");
assert.strictEqual(benchmarkResults.selected.benchmarks[9].score, 0.38);
assert.strictEqual(benchmarkResults.selected.benchmarks[9].fewshot, 4);
assert.strictEqual(benchmarkResults.kings[1].benchmarks[1].score, 0.2);
assert.strictEqual(benchmarkResults.kings[1].benchmarks[2].status, "pending");
assert.strictEqual(benchmarkResults.kings[1].benchmarks[2].fewshot, 10);
assert.deepStrictEqual(
    benchmarkResults.series[0].points.map((point) => [point.reignNumber, point.score]),
    [[6, 0.25], [7, 0.297344]]
);
assert.strictEqual(benchmarkResults.series[4].points[0].score, 0);
assert.doesNotThrow(() => dashboard.benchmarkPresentation({
    schema_version: "teutonic-king-benchmark-all-results.v1",
    kings: []
}));
assert.throws(
    () => dashboard.benchmarkPresentation({ schema_version: "wrong", kings: [] }),
    /unsupported benchmark results schema/
);
const benchmarkNames = ["BBH", "MMLU", "HellaSwag", "WinoGrande", "GSM8K", "PIQA", "ARC-C", "ARC-E", "GPQA Diamond", "MATH-500"];
assert.deepStrictEqual(
    dashboard.benchmarkPresentation({ schema_version: "teutonic-king-benchmark-all-results.v2", kings: [] }).series.map((series) => series.name),
    benchmarkNames
);
assert.strictEqual(dashboard.graphPointRadius(1, 1000, 0.75, 2.5), 2.5);
assert.strictEqual(dashboard.graphPointRadius(1000, 200, 0.75, 2.5), 0.75);
assert.ok(dashboard.graphPointRadius(200, 800, 0.75, 2.5) < 2.5);
const lossChart = dashboard.lossChartPresentation([
    { timestamp: "2026-08-18T12:02:00Z", avg_king_loss: 3.1, avg_challenger_loss: 2.7 },
    { timestamp: "2026-08-18T12:00:00Z", avg_king_loss: 2.6, avg_challenger_loss: 2.5 },
    { timestamp: "2026-08-18T12:01:00Z", avg_king_loss: 2.4, avg_challenger_loss: 2.3 },
    { timestamp: "2026-08-18T12:03:00Z", avg_king_loss: null, avg_challenger_loss: 2.1 }
]);
assert.strictEqual(lossChart.maximum, 2.5);
assert.deepStrictEqual(lossChart.points.map((point) => point.avg_challenger_loss), [2.5, 2.3]);
assert.deepStrictEqual(dashboard.lossChartPresentation([]), { points: [], maximum: null });
const datasetChanges = dashboard.datasetChangePresentation(
    [{ dataset_version: "a".repeat(64) }, { dataset_version: "b".repeat(64) }],
    [
        { config_version: "a".repeat(64), dataset_label: "mix-v1", eval_n: 1000, delta_threshold: 0.5, sources: [{ name: "fixture", proportion: 1, manifest_sha256: "c".repeat(64), total_tokens: 1000, total_shards: 1, sequence_length: 100 }] },
        { config_version: "b".repeat(64), dataset_label: "mix-v2", eval_n: 2000, delta_threshold: 0.5, sources: [{ name: "fixture", proportion: 1, manifest_sha256: "d".repeat(64), total_tokens: 2000, total_shards: 2, sequence_length: 100 }] }
    ]
);
assert.strictEqual(datasetChanges.length, 1);
assert.strictEqual(datasetChanges[0].index, 1);
assert.strictEqual(datasetChanges[0].fromLabel, "mix-v1");
assert.strictEqual(datasetChanges[0].toLabel, "mix-v2");
assert.ok(datasetChanges[0].changes.some((change) => change.includes("EVAL SAMPLES")));
assert.ok(datasetChanges[0].changes.some((change) => change.includes("fixture CONTENT")));
console.log("dashboard-v1 representative render states passed");
