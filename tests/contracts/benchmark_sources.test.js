"use strict";

const assert = require("node:assert/strict");
const dashboard = require("../../website/dashboard-v1.js");
const schema = "teutonic-king-benchmark-all-results.v2";
const names = ["bbh_cot", "MMLU_v2", "HellaSwag_v2", "WinoGrande_v2", "GSM8K", "PIQA", "ARC-C_v2", "ARC-E", "GPQA Diamond_v2", "MATH-500", "HumanEval Sampled n=3", "MBPP-Sanitized 3-shot n=3"];
const earlier = "2026-10-25T10:00:00Z", later = "2026-10-26T10:00:00Z";
function result(kingId, score, count = 12, overrides = {}) {
    return Object.assign({
        schema_version: "teutonic-king-benchmark-result.v1", generated_at: earlier,
        status: count === 12 ? "completed" : "running",
        model: { king_id: kingId, reign_number: 1, uid: 42, model_repo: "owner/" + kingId, crowned_at: earlier },
        benchmarks: names.slice(0, count).map(name => ({ name, status: "completed", metric: { value: score } }))
    }, overrides);
}
function publication(entries, generated = earlier) {
    return { schema_version: schema, generated_at: generated, kings: entries.map(entry => ({ king_id: entry.model.king_id, status: entry.status, updated_at: entry.generated_at, result: entry })) };
}

// Competition scopes distinguish equal reign IDs and identical models across mixes.
const scopes = ["main", "mix01", "mix02", "mix03", "mix04", "mix05", "mix06"];
let combined = dashboard.mergeBenchmarkSources(scopes.map((competition, index) => ({ competition, results: publication([result("same-id", .5 + index / 100)]) })));
let view = dashboard.benchmarkPresentation(combined);
assert.equal(view.kingCount, 7);
assert.equal(new Set(view.kings.map(king => king.key)).size, 7);
assert.equal(view.best.competition, "mix06");
assert.ok(Math.abs(view.best.average - .56) < 1e-12);
assert.equal(view.benchmarkResultCount, 84);
assert.equal(view.series[0].points.length, 7);

// MAIN and MIX reign numbers share one timeline ordered by date, not by reign.
const datedKing = (id, reign, crowned, generated) => result(id, .5, 12, {
    generated_at: generated || crowned,
    model: { king_id: id, reign_number: reign, crowned_at: crowned }
});
const timeline = dashboard.benchmarkPresentation(dashboard.mergeBenchmarkSources([
    { competition: "mix06", results: publication([datedKing("mix-early", 1, earlier)]) },
    { competition: "main", results: publication([datedKing("main-later", 49, later)]) },
    { competition: "mix01", results: publication([datedKing("mix-latest", 2, "2026-10-26T11:00:00Z")]) },
    { competition: "mix05", results: publication([datedKing("undated", 1, null, "2026-10-26T12:00:00Z")]) }
]));
assert.deepEqual(timeline.kings.map(king => king.kingId), ["undated", "mix-latest", "main-later", "mix-early"]);
assert.deepEqual(timeline.series[0].points.map(point => point.kingId), ["mix-early", "main-later", "mix-latest", "undated"]);
assert.deepEqual(timeline.series[0].points.map(point => point.timestamp), [earlier, later, "2026-10-26T11:00:00Z", "2026-10-26T12:00:00Z"]);

// A historical king's complete mean beats a current king and an easy partial run.
const historical = result("historical", .75);
const current = result("current", .5, 12, { model: { king_id: "current", reign_number: 2, is_current: true, crowned_at: later } });
const partial = result("partial", .99, 1);
view = dashboard.benchmarkPresentation(dashboard.mergeBenchmarkSources([{ competition: "main", results: publication([historical, current, partial]) }]));
assert.equal(view.selected.kingId, "historical");
assert.equal(view.kings.find(king => king.kingId === "partial").average, null);
assert.equal(view.kings.length, 3);
assert.equal(dashboard.benchmarkPresentation(combined, "main:same-id").selected.competition, "main");
const zero = dashboard.benchmarkPresentation(publication([result("zero", 0)]));
assert.equal(zero.best.average, 0);
assert.equal(zero.selected.scored, 12);

// Newer latest and per-king detail publications enrich the combined history.
const original = result("running", .4, 2);
const latest = result("running", .5, 1, { generated_at: later });
const detailed = result("running", .6, 3, { generated_at: "2026-10-26T11:00:00Z" });
detailed.benchmarks[0] = { name: names[0], status: "running", metric: null };
const source = {
    competition: "mix01", results: publication([original]),
    latest: { schema_version: "teutonic-king-benchmark-index.v1", generated_at: later, status: "running", current_king: latest.model, latest_result: latest },
    details: [{ king_id: "running", details_key: "competitions/mix01/kings/running/results.json", result: detailed }]
};
combined = dashboard.mergeBenchmarkSources([source]);
view = dashboard.benchmarkPresentation(combined);
assert.equal(view.kingCount, 1);
assert.deepEqual(view.selected.benchmarks.slice(0, 3).map(row => row.score), [.5, .6, .6]);
assert.equal(view.selected.current, true);
assert.equal(view.selected.average, null);
assert.deepEqual(dashboard.benchmarkDetailRequests(source), [{ kingId: "running", key: "competitions/mix01/kings/running/results.json" }]);
assert.equal(source.results.kings[0].result.benchmarks[0].metric.value, .4, "Merging must not mutate stored publications");

// A running king can appear in latest.json before the combined file lists it.
const waiting = {
    competition: "mix06", results: publication([result("old", .5)]),
    latest: { generated_at: later, status: "running", current_king: { king_id: "new", reign_number: 2, is_current: true } }
};
combined = dashboard.mergeBenchmarkSources([waiting]);
view = dashboard.benchmarkPresentation(combined);
assert.equal(view.kingCount, 2);
assert.equal(view.kings.find(king => king.kingId === "new").current, true);
assert.equal(view.kings.find(king => king.kingId === "new").scored, 0);
assert.equal(view.best.kingId, "old");
assert.deepEqual(dashboard.benchmarkDetailRequests(waiting), [{ kingId: "new", key: "competitions/mix06/kings/new/results.json" }]);

// A stale latest publication cannot restore an old king's current flag or scores.
const fresh = result("fresh", .7, 12, { generated_at: later, model: { king_id: "fresh", is_current: true } });
const old = result("old", .4);
const stale = result("old", .3, 1, { model: { king_id: "old", is_current: true } });
view = dashboard.benchmarkPresentation(dashboard.mergeBenchmarkSources([{
    competition: "main", results: publication([fresh, old], later),
    latest: { generated_at: earlier, current_king: stale.model, latest_result: stale }
}]));
assert.deepEqual(view.kings.filter(king => king.current).map(king => king.kingId), ["fresh"]);
assert.equal(view.kings.find(king => king.kingId === "old").benchmarks[0].score, .4);

assert.equal(dashboard.benchmarkDetailsKey("mix01", "running", "competitions/mix02/kings/running/results.json"), null);
assert.equal(dashboard.benchmarkDetailsKey("main", "../private"), null);
assert.equal(dashboard.benchmarkDetailsKey("main", "king"), "king-benchmark-daily/kings/king/results.json");
assert.throws(() => dashboard.mergeBenchmarkSources([{ competition: "mix01", details: [{ king_id: "wrong", result: result("other", .5) }] }]), /does not match/);
assert.throws(() => dashboard.mergeBenchmarkSources([{ competition: "mix01", results: { schema_version: "invalid", kings: [] } }]), /unsupported benchmark/);

console.log("Benchmark sources: seven scopes, complete averages, running updates, stale publications, and detail identities passed.");
