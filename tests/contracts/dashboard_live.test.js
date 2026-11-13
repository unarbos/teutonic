"use strict";
const assert = require("node:assert/strict");
const api = require("../../website/dashboard-v1.js");
const stamp = "2026-11-12T20:36:20Z", now = Date.parse(stamp);
const identity = {challenge_id: "75f8c3e10931a158", model_identity: "hidden_until_promotion"};
function snapshot() {
    return {schema_version: 1, generated_at: stamp, netuid: 3, generation: "test", source_watermark: 10,
        competitions: [
            {key: "main", name: "Teutonic", queue: [identity], current_eval: null, service_status: {}},
            {key: "mix01", name: "mix01", queue: [], current_eval: identity, service_status: {}}
        ]};
}
const base = [{key: "main", failed: true, dashboard: {
    chain: {netuid: 3, generation: "test"}, generated_at: "2026-11-12T20:20:00Z",
    source_watermark: 5, queue: [], current_eval: identity, history: [{old: true}]
}}];
const merged = api.liveEvaluationSources(base, snapshot(), now, false);
assert.equal(merged[0].dashboard.queue.length, 1);
assert.equal(merged[0].dashboard.current_eval, null); // Clear stale active MAIN.
assert.equal(merged[0].failed, false); // Slow history failure does not poison live status.
assert.equal(merged[1].dashboard.current_eval, identity); // Works before competition index loads.
assert.deepEqual(base[0].dashboard.queue, []); // History payload is not mutated.
assert.equal(merged[0].dashboard.history, base[0].dashboard.history);
assert.equal(api.liveEvaluationSources(base, snapshot(), now + 31000, false)[0].failed, true);
assert.equal(api.liveEvaluationSources(base, snapshot(), now, true)[0].failed, true);
const empty = snapshot(); empty.competitions.forEach(item => {item.queue = []; item.current_eval = null;});
assert(api.liveEvaluationSources(base, empty, now, false).every(item => item.dashboard.current_eval === null && !item.dashboard.queue.length));
const newer = structuredClone(base); newer[0].dashboard.generated_at = "2026-11-12T20:37:00Z"; newer[0].dashboard.source_watermark = 11;
assert.equal(api.liveEvaluationSources(newer, snapshot(), now, false)[0].dashboard.current_eval, null);
const wrong = snapshot(); wrong.generation = "other";
assert.equal(api.liveEvaluationSources(base, wrong, now, false), base);
assert.equal(api.liveEvaluationSources(base, null, now, true), base);
assert.equal(api.liveEvaluationSources(base, snapshot(), now - 60000, false), base);
const invalid = snapshot(); invalid.competitions.push(invalid.competitions[0]);
assert.throws(() => api.validateLive(invalid));
const doubleActive = snapshot(); doubleActive.competitions[0].current_eval = identity;
assert.throws(() => api.validateLive(doubleActive));
console.log("dashboard live contract tests passed");
