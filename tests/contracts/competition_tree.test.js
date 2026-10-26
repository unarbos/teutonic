"use strict";

const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const path = require("node:path");
const { buildGraph, compareMixes } = require("../../website/competition-tree.js");

const config = JSON.parse(execFileSync("python3", ["-c", [
    "import json, tomllib",
    "with open('chain.toml', 'rb') as source: config = tomllib.load(source)",
    "print(json.dumps({key: [[config['categories'][name]['dataset'], name, weight] for name, weight in value['categories'].items()] for key, value in config['competitions'].items()}))"
].join("\n")], { cwd: path.resolve(__dirname, "../.."), encoding: "utf8" }));

function node(key, entries) {
    const total = entries.reduce((sum, entry) => sum + entry[2], 0);
    const weights = new Map(entries.map(([dataset, category, weight]) => [dataset + "\0" + category, weight / total]));
    const categories = new Map(entries.map(([dataset, name]) => [dataset + "\0" + name, { dataset, name }]));
    return { key, weights, categories };
}

// Every real overlap must survive, including partial overlaps across branches
// of the old containment tree. Source dataset and category jointly identify data.
const nodes = Object.entries(config).map(([key, entries]) => node(key, entries));
const links = buildGraph(nodes);
assert.equal(links.length, 21);
const byKey = new Map(nodes.map(value => [value.key, value]));
const seen = new Set();
for (const link of links) {
    const a = byKey.get(link.from), b = byKey.get(link.to);
    const key = [link.from, link.to].sort().join(":");
    assert.notEqual(link.from, link.to);
    assert.ok(!seen.has(key), "Only one undirected connection per pair");
    seen.add(key);
    assert.deepEqual(link.shared, [...a.categories.keys()].filter(category => b.categories.has(category)));
    assert.deepEqual(link.onlyFrom, [...a.categories.keys()].filter(category => !b.categories.has(category)));
    assert.deepEqual(link.onlyTo, [...b.categories.keys()].filter(category => !a.categories.has(category)));
    const expectedWeight = [...a.weights].reduce((sum, [category, weight]) => sum + Math.min(weight, b.weights.get(category) || 0), 0);
    assert.ok(Math.abs(link.weight - expectedWeight) < 1e-12);
    assert.ok(Math.abs(link.weight - compareMixes(b, a).weight) < 1e-12);
}
const main = byKey.get("main");
for (const [key, shared, added, missing] of [["mix06", 24, 1, 0], ["mix01", 18, 1, 6], ["mix05", 7, 1, 17]]) {
    const comparison = compareMixes(byKey.get(key), main);
    assert.equal(comparison.shared.length, shared);
    assert.equal(comparison.onlyFrom.length, added);
    assert.equal(comparison.onlyTo.length, missing);
    assert.equal(byKey.get(key).categories.get(comparison.onlyFrom[0]).name, "dclm-baseline-1.0");
}
assert.ok(Math.abs(byKey.get("mix01").weights.get("dclm-baseline-1.0\0dclm-baseline-1.0") - 0.1) < 1e-12);
assert.ok(Math.abs(byKey.get("mix05").weights.get("dclm-baseline-1.0\0dclm-baseline-1.0") - 0.4) < 1e-12);
const mix05mix06 = compareMixes(byKey.get("mix05"), byKey.get("mix06"));
assert.equal(mix05mix06.shared.length, 8);
assert.ok(Math.abs(mix05mix06.weight - 0.46359867374521413) < 1e-12);

// Partial and equal sets both connect. Disjoint categories and identically
// named categories from different original datasets never create false links.
const entries = (...categories) => categories.map(category => ["dataset", category, 1]);
const partial = buildGraph([
    node("a", entries("one", "two")), node("b", entries("two", "three"))
]);
assert.equal(partial.length, 1);
assert.deepEqual(partial[0].shared, ["dataset\0two"]);
assert.equal(partial[0].weight, 0.5);
assert.equal(buildGraph([
    node("a", entries("one")), node("b", entries("one"))
]).length, 1);
assert.deepEqual(buildGraph([
    node("a", entries("one")), node("b", entries("two"))
]), []);
assert.deepEqual(buildGraph([
    node("a", [["source-a", "one", 1]]),
    node("b", [["source-b", "one", 1], ["source-b", "two", 1]])
]), []);
assert.deepEqual(buildGraph([]), []);
assert.deepEqual(buildGraph([node("a", entries("one"))]), []);
console.log("Competition graph: 21 real overlaps, MAIN/DCLM relationships, unique categories, and exact source identities passed.");
