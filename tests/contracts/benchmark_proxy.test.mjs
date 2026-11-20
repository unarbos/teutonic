import assert from "node:assert/strict";
import worker from "../../scripts/cloudflare/teutonic-proxy/worker.mjs";

const origin = "https://pub-c982d552b8044578b4a79e653700ec73.r2.dev";
const originalFetch = globalThis.fetch;
const calls = [];
globalThis.fetch = async (url, options) => {
    calls.push({url, options});
    return new Response('{"status":"running"}', {headers: {"Content-Type": "application/json"}});
};
try {
    for (const scope of ["king-benchmark-daily", "competitions/main2", ...Array.from({length: 9}, (_, i) => "competitions/mix0" + (i + 1))]) {
        for (const feed of ["all-kings/results.json", "all-kings/latest.json", "kings/reign-000001/results.json"]) {
            const path = "/benchmarks/" + scope + "/" + feed;
            for (const method of ["GET", "HEAD"]) {
                const response = await worker.fetch(new Request("https://teutonic.ai" + path, {method}));
                assert.equal(calls.at(-1).url, origin + "/" + scope + "/" + feed);
                assert.equal(calls.at(-1).options.method, method);
                assert.equal(response.status, 200);
                assert.match(response.headers.get("Cache-Control"), /no-cache/);
            }
            const count = calls.length;
            assert.equal((await worker.fetch(new Request("https://teutonic.ai" + path, {method: "POST"}))).status, 405);
            assert.equal(calls.length, count);
        }
    }
    for (const path of ["/benchmarks/competitions/mix10/all-kings/latest.json", "/benchmarks/competitions/main2/private.json"]) {
        await worker.fetch(new Request("https://teutonic.ai" + path));
        assert.ok(!calls.at(-1).url.startsWith(origin));
    }
} finally {
    globalThis.fetch = originalFetch;
}
console.log("Benchmark proxy: all competition feeds, detail routes, cache policy and method restrictions passed.");
