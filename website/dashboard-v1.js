(function(root, factory) {
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    root.TeutonicDashboardV1 = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function() {
    "use strict";

    var HIDDEN = "Hidden until promotion";
    var BENCHMARK_SPECS = [
        { name: "BBH CoT", sourceName: "bbh_cot", fewshot: 3 },
        { name: "MMLU", fewshot: 5 },
        { name: "HellaSwag", fewshot: 10 },
        { name: "WinoGrande", fewshot: 5 },
        { name: "GSM8K", fewshot: 4 },
        { name: "PIQA", fewshot: 0 },
        { name: "ARC-C", fewshot: 25 },
        { name: "ARC-E", fewshot: 0 },
        { name: "GPQA Diamond", fewshot: 5 },
        { name: "MATH-500", fewshot: 4 },
        { name: "HumanEval-Sanitized", sourceName: "HumanEval Sampled n=3", fewshot: 0 },
        { name: "MBPP-Sanitized", sourceName: "MBPP-Sanitized 3-shot n=3", fewshot: 3 }
    ];

    function benchmarkKey(name) {
        return String(name || "").replace(/_v\d+$/i, "").toLowerCase();
    }

    function walkFinite(value, path) {
        if (typeof value === "number" && !Number.isFinite(value)) {
            throw new Error("non-finite dashboard number at " + path);
        }
        if (Array.isArray(value)) {
            value.forEach(function(item, index) { walkFinite(item, path + "[" + index + "]"); });
        } else if (value && typeof value === "object") {
            Object.keys(value).forEach(function(key) { walkFinite(value[key], path + "." + key); });
        }
    }

    function validate(payload) {
        if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
            throw new Error("dashboard payload must be an object");
        }
        if (payload.schema_version !== 1) throw new Error("unsupported dashboard schema");
        ["queue", "history", "king_chain", "dataset_versions"].forEach(function(name) {
            if (!Array.isArray(payload[name])) throw new Error("dashboard " + name + " must be an array");
        });
        ["chain", "stats", "king_payout", "weight_status", "service_status"].forEach(function(name) {
            if (!payload[name] || typeof payload[name] !== "object" || Array.isArray(payload[name])) {
                throw new Error("dashboard " + name + " must be an object");
            }
        });
        walkFinite(payload, "$");
        return payload;
    }

    function identityLabel(record) {
        return record && record.model_identity === "hidden_until_promotion" ? HIDDEN : null;
    }

    function presentation(payload) {
        validate(payload);
        var current = payload.current_eval;
        var queued = payload.queue.length ? payload.queue[0] : null;
        var services = payload.service_status;
        return {
            idle: !current && !queued,
            duelIdentity: identityLabel(current || queued),
            queueIdentity: payload.queue.map(function(item) { return identityLabel(item); }),
            historyIdentity: payload.history.map(function(item) {
                return identityLabel(item) || item.challenger_repo || "Public model";
            }),
            historyEmpty: payload.history.length === 0,
            queueEmpty: payload.queue.length === 0,
            health: services.overall,
            degraded: services.overall !== "healthy",
            marketStale: !!(payload.market && payload.market.stale),
            marketAvailable: !!payload.market
        };
    }

    function finiteNumber(value, fallback) {
        var number = Number(value);
        return Number.isFinite(number) ? number : fallback;
    }

    function graphPointRadius(pointCount, plotWidth, minimum, maximum) {
        var count = Math.max(1, Math.floor(finiteNumber(pointCount, 1)));
        var width = Math.max(1, finiteNumber(plotWidth, 1));
        var minRadius = Math.max(0.5, finiteNumber(minimum, 0.75));
        var maxRadius = Math.max(minRadius, finiteNumber(maximum, 2.5));
        if (count <= 1) return maxRadius;
        return Math.max(minRadius, Math.min(maxRadius, width / (count - 1) * 0.28));
    }

    function lossChartPresentation(history) {
        function loss(value) {
            return value == null || value === "" ? null : finiteNumber(value, null);
        }
        var points = (Array.isArray(history) ? history : []).filter(function(item) {
            return item
                && loss(item.avg_king_loss) != null
                && loss(item.avg_challenger_loss) != null;
        }).sort(function(a, b) {
            return new Date(a.timestamp || 0) - new Date(b.timestamp || 0);
        });
        if (!points.length) return { points: [], maximum: null };
        var maximum = loss(points[0].avg_challenger_loss);
        if (maximum == null || maximum <= 0) return { points: [], maximum: null };
        return {
            points: points.filter(function(item) {
                return loss(item.avg_challenger_loss) <= maximum;
            }),
            maximum: maximum
        };
    }

    function datasetChangePresentation(history, datasetVersions) {
        var configs = {};
        (Array.isArray(datasetVersions) ? datasetVersions : []).forEach(function(config) {
            if (config && typeof config.config_version === "string") {
                configs[config.config_version] = config;
            }
        });
        function shortVersion(value) { return String(value || "").slice(0, 8) || "unknown"; }
        function numberLabel(value) {
            var number = finiteNumber(value, null);
            return number == null ? "--" : number.toLocaleString();
        }
        function percentLabel(value) {
            var number = finiteNumber(value, null);
            return number == null ? "--" : (number * 100).toFixed(1).replace(/\.0$/, "") + "%";
        }
        function valueLabel(value) { return value == null || value === "" ? "--" : String(value); }
        function changesBetween(previous, next) {
            if (!previous || !next) return ["CONFIGURATION DETAILS UNAVAILABLE"];
            var changes = [];
            if (previous.dataset_label !== next.dataset_label) changes.push("DATASET MIX: " + valueLabel(previous.dataset_label) + " → " + valueLabel(next.dataset_label));
            if (finiteNumber(previous.eval_n, null) !== finiteNumber(next.eval_n, null)) changes.push("EVAL SAMPLES: " + numberLabel(previous.eval_n) + " → " + numberLabel(next.eval_n));
            if (finiteNumber(previous.delta_threshold, null) !== finiteNumber(next.delta_threshold, null)) changes.push("DELTA THRESHOLD: " + valueLabel(previous.delta_threshold) + " → " + valueLabel(next.delta_threshold));
            var beforeSources = {}, afterSources = {};
            (Array.isArray(previous.sources) ? previous.sources : []).forEach(function(source) { if (source && source.name) beforeSources[source.name] = source; });
            (Array.isArray(next.sources) ? next.sources : []).forEach(function(source) { if (source && source.name) afterSources[source.name] = source; });
            Object.keys(beforeSources).sort().forEach(function(name) { if (!afterSources[name]) changes.push("SOURCE REMOVED: " + name); });
            Object.keys(afterSources).sort().forEach(function(name) {
                var before = beforeSources[name], after = afterSources[name];
                if (!before) { changes.push("SOURCE ADDED: " + name + " (" + percentLabel(after.proportion) + ")"); return; }
                if (finiteNumber(before.proportion, null) !== finiteNumber(after.proportion, null)) changes.push(name + " WEIGHT: " + percentLabel(before.proportion) + " → " + percentLabel(after.proportion));
                if (before.manifest_sha256 !== after.manifest_sha256) changes.push(name + " CONTENT: " + shortVersion(before.manifest_sha256) + " → " + shortVersion(after.manifest_sha256));
                if (finiteNumber(before.total_tokens, null) !== finiteNumber(after.total_tokens, null)) changes.push(name + " TOKENS: " + numberLabel(before.total_tokens) + " → " + numberLabel(after.total_tokens));
                if (finiteNumber(before.total_shards, null) !== finiteNumber(after.total_shards, null)) changes.push(name + " SHARDS: " + numberLabel(before.total_shards) + " → " + numberLabel(after.total_shards));
                if (finiteNumber(before.sequence_length, null) !== finiteNumber(after.sequence_length, null)) changes.push(name + " SEQUENCE LENGTH: " + numberLabel(before.sequence_length) + " → " + numberLabel(after.sequence_length));
                ["source_repo", "tokenizer", "dtype", "tokenization_mode"].forEach(function(field) {
                    if (valueLabel(before[field]) !== valueLabel(after[field])) changes.push(name + " " + field.replaceAll("_", " ").toUpperCase() + ": " + valueLabel(before[field]) + " → " + valueLabel(after[field]));
                });
            });
            return changes.length ? changes : ["CONFIGURATION HASH CHANGED; PUBLISHED SETTINGS ARE OTHERWISE IDENTICAL"];
        }
        var rows = Array.isArray(history) ? history : [], events = [];
        for (var index = 1; index < rows.length; index++) {
            var fromVersion = String(rows[index - 1].dataset_version || ""), toVersion = String(rows[index].dataset_version || "");
            if (!fromVersion || !toVersion || fromVersion === toVersion) continue;
            events.push({
                index: index,
                fromVersion: fromVersion,
                toVersion: toVersion,
                fromLabel: configs[fromVersion] ? configs[fromVersion].dataset_label : shortVersion(fromVersion),
                toLabel: configs[toVersion] ? configs[toVersion].dataset_label : shortVersion(toVersion),
                changes: changesBetween(configs[fromVersion], configs[toVersion])
            });
        }
        return events;
    }

    function currentEvaluationPresentation(record) {
        record = record || {};
        function optionalNumber(value) {
            return value == null || value === "" ? null : finiteNumber(value, null);
        }
        var lcb = optionalNumber(record.provisional_lcb);
        var muHat = optionalNumber(record.provisional_mu_hat);
        var threshold = optionalNumber(record.delta_threshold);
        var sequences = optionalNumber(record.provisional_n_sequences);
        var bootstraps = optionalNumber(record.provisional_n_bootstrap);
        return {
            available: lcb != null,
            lcb: lcb,
            muHat: muHat,
            threshold: threshold,
            sequences: sequences == null ? null : Math.max(0, Math.floor(sequences)),
            bootstraps: bootstraps == null ? null : Math.max(0, Math.floor(bootstraps)),
            clearsThreshold: lcb == null || threshold == null ? null : lcb > threshold
        };
    }

    function historyPresentation(history, showErrors) {
        if (!Array.isArray(history)) throw new Error("dashboard history must be an array");
        var errorCount = history.filter(function(item) {
            return item && item.verdict === "error";
        }).length;
        return {
            rows: showErrors ? history.slice() : history.filter(function(item) {
                return !item || item.verdict !== "error";
            }),
            errorCount: errorCount
        };
    }

    function verdictLabel(verdict) {
        var value = String(verdict || "").toLowerCase();
        if (value === "accepted") return "CHALLENGER";
        if (value === "rejected") return "KING";
        return value ? value.toUpperCase() : "--";
    }

    function evaluationHistoryMetricsPresentation(record) {
        record = record || {};
        function optionalCount(value) {
            if (value == null || value === "") return null;
            var count = finiteNumber(value, null);
            return count == null ? null : Math.max(0, Math.floor(count));
        }
        var evaluated = optionalCount(record.n_sequences_evaluated);
        var planned = optionalCount(record.n_sequences);
        var completed = evaluated == null ? planned : evaluated;
        var hasSamples = completed != null;
        return {
            samples: !hasSamples ? "--" : evaluated != null && planned != null && evaluated !== planned
                ? evaluated + " / " + planned : String(completed),
            samplesTitle: !hasSamples ? "Sample count unavailable" : evaluated != null && planned != null
                ? evaluated + " of " + planned + " samples evaluated" : completed + " samples evaluated",
            earlyStopped: record.early_stopped === true,
            earlyStopLabel: record.early_stopped === true ? "YES" : hasSamples ? "NO" : "--"
        };
    }

    function sourceScoresPresentation(record) {
        var rawScores = record && Array.isArray(record.source_scores) ? record.source_scores : [];
        var rows = rawScores.map(function(raw) {
            raw = raw || {};
            var source = typeof raw.source === "string" ? raw.source.trim() : "";
            var nSequences = raw.n_sequences == null ? null : finiteNumber(raw.n_sequences, null);
            var kingLoss = raw.avg_king_loss == null ? null : finiteNumber(raw.avg_king_loss, null);
            var challengerLoss = raw.avg_challenger_loss == null ? null : finiteNumber(raw.avg_challenger_loss, null);
            var muHat = raw.mu_hat == null ? null : finiteNumber(raw.mu_hat, null);
            if (!source || nSequences == null || kingLoss == null || challengerLoss == null || muHat == null) return null;
            return {
                source: source,
                nSequences: Math.max(0, Math.floor(nSequences)),
                kingLoss: kingLoss,
                challengerLoss: challengerLoss,
                muHat: muHat
            };
        }).filter(Boolean).sort(function(a, b) { return a.source.localeCompare(b.source); });
        return { rows: rows, count: rows.length };
    }

    function categoryScoresPresentation(record) {
        return sourceScoresPresentation({ source_scores: record && record.category_scores });
    }

    function taoMarketCapHotkeyUrl(hotkey) {
        var address = String(hotkey || "").trim();
        if (!address) return "";
        return "https://taomarketcap.com/hotkey/" + encodeURIComponent(address) + "/metagraph";
    }

    function taoMarketCapColdkeyUrl(coldkey) {
        var address = String(coldkey || "").trim();
        if (!address) return "";
        return "https://taomarketcap.com/coldkey/" + encodeURIComponent(address);
    }

    function shardPresentation(record) {
        var rawGroups = record && Array.isArray(record.shards_used) ? record.shards_used : [];
        var groups = rawGroups.map(function(group) {
            var source = group && typeof group.source === "string" && group.source.trim()
                ? group.source.trim() : "dataset";
            var rawNames = group && Array.isArray(group.names) ? group.names : [];
            var names = [], seen = new Set();
            rawNames.forEach(function(value) {
                if (typeof value !== "string" || !value.trim()) return;
                var name = value.trim();
                if (!seen.has(name)) { seen.add(name); names.push(name); }
            });
            return { source: source, names: names };
        }).filter(function(group) { return group.names.length > 0; });
        return {
            groups: groups,
            count: groups.reduce(function(total, group) { return total + group.names.length; }, 0)
        };
    }

    function uploadFailurePresentation(record) {
        if (!record || !record.upload_id) return null;
        return {
            registration: "UID " + String(record.uid == null ? "--" : record.uid)
                + " · " + String(record.registration_state || "unknown").toUpperCase(),
            uploadId: String(record.upload_id),
            uploadState: String(record.upload_state || "verification_failed"),
            failureCode: String(record.error_code || "verification_failed")
        };
    }

    function decisionPresentation(record) {
        record = record || {};
        var verdict = String(record.verdict || "").toLowerCase();
        var lcb = finiteNumber(record.lcb, null);
        var threshold = finiteNumber(
            record.delta_threshold == null ? record.delta : record.delta_threshold,
            null
        );
        var won = record.accepted === true || verdict === "accepted" || verdict === "challenger";
        var lost = record.accepted === false || verdict === "rejected" || verdict === "king";

        if (verdict === "error") {
            return {
                kind: "error",
                label: "ERROR REASON",
                summary: record.error_message || "The evaluation did not produce a verdict.",
                detail: "No win or loss was recorded."
            };
        }
        if (won) {
            return {
                kind: "win",
                label: "WIN REASON",
                summary: lcb == null || threshold == null
                    ? "The challenger cleared the promotion policy."
                    : "LCB " + lcb.toFixed(6) + " > REQUIRED " + threshold.toFixed(6)
                        + " · MARGIN +" + (lcb - threshold).toFixed(6),
                detail: "The confidence-adjusted improvement was high enough to replace the king."
            };
        }
        if (lost) {
            return {
                kind: "loss",
                label: "LOSS REASON",
                summary: lcb == null || threshold == null
                    ? "The challenger did not clear the promotion policy."
                    : "LCB " + lcb.toFixed(6) + " ≤ REQUIRED " + threshold.toFixed(6)
                        + " · SHORTFALL " + Math.max(0, threshold - lcb).toFixed(6),
                detail: "The measured improvement was not confident enough to replace the king."
            };
        }
        return {
            kind: "unknown",
            label: "DECISION",
            summary: "A final win or loss reason is not available.",
            detail: ""
        };
    }

    function datasetPresentation(manifest) {
        if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
            throw new Error("dataset manifest must be an object");
        }
        if (!Array.isArray(manifest.sources)) {
            throw new Error("dataset manifest sources must be an array");
        }
        var sources = manifest.sources.filter(function(source) {
            return source && source.enabled !== false;
        });
        var rawWeights = sources.map(function(source) {
            return Math.max(0, finiteNumber(
                source.proportion == null ? source.weight : source.proportion,
                0
            ));
        });
        var weightTotal = rawWeights.reduce(function(total, weight) { return total + weight; }, 0);
        var defaultWeight = sources.length ? 1 / sources.length : 0;
        var evalN = Math.max(0, finiteNumber(manifest.eval_n || manifest.default_eval_sequences, 0));
        var rows = sources.map(function(source, index) {
            var weight = weightTotal > 0 ? rawWeights[index] / weightTotal : defaultWeight;
            var sequenceLength = Math.max(0, finiteNumber(
                source.sequence_length || source.seq_len || manifest.sequence_length,
                0
            ));
            var totalTokens = Math.max(0, finiteNumber(source.total_tokens, 0));
            var totalShards = Math.max(0, finiteNumber(
                source.total_shards,
                0
            ));
            var sequences = Math.max(0, finiteNumber(
                source.estimated_sequences,
                sequenceLength ? Math.floor(totalTokens / sequenceLength) : 0
            ));
            var evalSequences = evalN ? Math.round(evalN * weight) : 0;
            var evalTokens = evalSequences * sequenceLength;
            return {
                name: source.name || "dataset",
                manifestUrl: source.manifest_url || "",
                manifestSha256: source.manifest_sha256 || "",
                weight: finiteNumber(source.proportion == null ? source.weight : source.proportion, 0),
                normalizedWeight: weight,
                totalTokens: totalTokens,
                totalShards: totalShards,
                sequences: sequences,
                sequenceLength: sequenceLength,
                evalSequences: evalSequences,
                evalTokens: evalTokens,
                sampleRate: totalTokens ? evalTokens / totalTokens * 100 : 0,
                source: source.source_repo || source.source || "",
                tokenizer: source.tokenizer || "",
                dtype: source.dtype || "",
                tokenizationMode: source.tokenization_mode || "",
                metadataLoaded: Boolean(totalTokens || totalShards || sequenceLength || source.source_repo)
            };
        });
        var totalTokens = rows.reduce(function(total, row) { return total + row.totalTokens; }, 0);
        var totalShards = rows.reduce(function(total, row) { return total + row.totalShards; }, 0);
        var totalSequences = rows.reduce(function(total, row) { return total + row.sequences; }, 0);
        var sequenceLengths = rows.map(function(row) { return row.sequenceLength; }).filter(Boolean);
        var sharedSequenceLength = sequenceLengths.length && sequenceLengths.every(function(value) {
            return value === sequenceLengths[0];
        }) ? sequenceLengths[0] : 0;
        var tokenizers = rows.map(function(row) { return row.tokenizer; }).filter(Boolean);
        var sharedTokenizer = tokenizers.length && tokenizers.every(function(value) {
            return value === tokenizers[0];
        }) ? tokenizers[0] : "";
        return {
            label: manifest.dataset_label || manifest.name || "dataset mix",
            generatedAt: manifest.generated_at || manifest.updated || "",
            evalN: evalN,
            rows: rows,
            totalTokens: totalTokens,
            totalShards: totalShards,
            totalSequences: totalSequences,
            sequenceLength: sharedSequenceLength,
            evalTokens: rows.reduce(function(total, row) { return total + row.evalTokens; }, 0),
            tokenizer: sharedTokenizer
        };
    }

    function mainDatasetPresentation(manifest) {
        if (!manifest || !Array.isArray(manifest.sources)) return datasetPresentation(manifest);
        var groups = new Map();
        manifest.sources.forEach(function(source) {
            if (!source || source.enabled === false) return;
            var name = source.dataset || source.name;
            if (!groups.has(name)) {
                groups.set(name, Object.assign({}, source, {
                    name: name, proportion: 0, total_tokens: 0, total_shards: 0,
                    estimated_sequences: 0
                }));
            }
            var group = groups.get(name);
            group.proportion += finiteNumber(source.proportion == null ? source.weight : source.proportion, 0);
            group.total_tokens += finiteNumber(source.total_tokens, 0);
            group.total_shards += finiteNumber(source.total_shards, 0);
            var length = source.sequence_length || source.seq_len || manifest.sequence_length;
            group.estimated_sequences += finiteNumber(source.estimated_sequences,
                length ? Math.floor(finiteNumber(source.total_tokens, 0) / length) : 0);
        });
        var order = ["pretrain-behaviors", "math-reasoning", "txt360-v2", "code-reasoning", "sft-reasoning"];
        var sources = Array.from(groups.values()).sort(function(a, b) {
            var ai = order.indexOf(a.name), bi = order.indexOf(b.name);
            return (ai < 0 ? order.length : ai) - (bi < 0 ? order.length : bi);
        });
        return datasetPresentation(Object.assign({}, manifest, { sources: sources }));
    }

    function benchmarkTimestamp(value) {
        var time = Date.parse(value || "");
        return Number.isFinite(time) ? time : 0;
    }

    function benchmarkDetailsKey(competition, kingId, key) {
        if (!/^(main|mix0[1-6])$/.test(competition || "") || !/^[A-Za-z0-9_-]+$/.test(kingId || "")) return null;
        var prefix = competition === "main" ? "king-benchmark-daily" : "competitions/" + competition;
        var expected = prefix + "/kings/" + kingId + "/results.json";
        return !key || key === expected ? expected : null;
    }

    function benchmarkDetailRequests(source) {
        var requests = new Map(), competition = source.competition;
        function add(kingId, key) {
            var path = benchmarkDetailsKey(competition, kingId, key);
            if (path) requests.set(kingId, { kingId: kingId, key: path });
        }
        ((source.results || {}).kings || []).forEach(function(item) {
            if (!item || typeof item !== "object") return;
            if (String(item.status || (item.result || {}).status || "pending").toLowerCase() !== "completed") add(item.king_id || ((item.result || {}).model || {}).king_id, item.details_key);
        });
        var latest = source.latest || {}, model = latest.current_king || {}, result = latest.latest_result || {};
        if (model.king_id && String(latest.status || "pending").toLowerCase() !== "completed") add(model.king_id);
        if (result.model && String(result.status || "pending").toLowerCase() !== "completed") add(result.model.king_id);
        return Array.from(requests.values());
    }

    function mergeBenchmarkSources(sources) {
        var merged = new Map(), generatedAt = null;
        sources.forEach(function(source) {
            var competition = source.competition, payload = source.results || {}, latest = source.latest || {};
            if (!/^(main|mix0[1-6])$/.test(competition || "")) throw new Error("unsupported benchmark competition");
            if (source.results && (["teutonic-king-benchmark-all-results.v1", "teutonic-king-benchmark-all-results.v2"].indexOf(payload.schema_version) === -1 || !Array.isArray(payload.kings))) throw new Error("unsupported benchmark results schema");
            var candidates = (payload.kings || []).filter(function(item) { return item && typeof item === "object"; });
            var latestResult = latest.latest_result;
            if (latestResult && latestResult.model && latestResult.model.king_id) candidates.unshift({ king_id: latestResult.model.king_id, updated_at: latestResult.generated_at || latest.generated_at, status: latestResult.status, result: latestResult });
            var current = latest.current_king;
            if (current && current.king_id && !candidates.some(function(item) { return (item.king_id || ((item.result || {}).model || {}).king_id) === current.king_id; })) candidates.push({ king_id: current.king_id, updated_at: latest.generated_at, status: latest.status || "pending", result: { model: current, benchmarks: [] } });
            (source.details || []).forEach(function(item) {
                var result = item.result || {};
                if (!result.model || result.model.king_id !== item.king_id) throw new Error("benchmark detail king does not match request");
                candidates.push({ king_id: item.king_id, details_key: item.details_key, updated_at: result.generated_at, status: result.status, result: result });
            });
            [payload.generated_at, latest.generated_at].forEach(function(value) { if (benchmarkTimestamp(value) > benchmarkTimestamp(generatedAt)) generatedAt = value; });
            candidates.sort(function(a, b) { return benchmarkTimestamp(a.updated_at || (a.result || {}).generated_at) - benchmarkTimestamp(b.updated_at || (b.result || {}).generated_at); });
            candidates.forEach(function(item) {
                var result = item.result || {}, kingId = item.king_id || (result.model || {}).king_id;
                if (!kingId) return;
                var identity = competition + ":" + kingId, previous = merged.get(identity) || {}, oldResult = previous.result || {};
                var benchmarks = new Map();
                (oldResult.benchmarks || []).forEach(function(row) { if (row && typeof row.name === "string") benchmarks.set(benchmarkKey(row.name), row); });
                (result.benchmarks || []).forEach(function(row) {
                    if (!row || typeof row.name !== "string") return;
                    var name = benchmarkKey(row.name), old = benchmarks.get(name);
                    var oldScore = old && old.metric && old.metric.value != null && finiteNumber(old.metric.value, null) != null;
                    var newScore = row.metric && row.metric.value != null && finiteNumber(row.metric.value, null) != null;
                    if (oldScore && (!newScore || String(old.status).toLowerCase() === "completed" && String(row.status).toLowerCase() !== "completed")) return;
                    benchmarks.set(name, Object.assign({}, old, row));
                });
                var model = Object.assign({}, oldResult.model, result.model);
                model.competition_key = competition;
                var next = Object.assign({}, previous, item, { competition: competition, king_id: kingId });
                next.details_key = benchmarkDetailsKey(competition, kingId, item.details_key || previous.details_key);
                next.result = Object.assign({}, oldResult, result, { model: model, benchmarks: Array.from(benchmarks.values()) });
                merged.set(identity, next);
                if (benchmarkTimestamp(next.updated_at) > benchmarkTimestamp(generatedAt)) generatedAt = next.updated_at;
            });
            var publicationCurrent = (payload.kings || []).find(function(item) { return item && ((item.result || {}).model || {}).is_current === true; });
            var currentId = current && benchmarkTimestamp(latest.generated_at) >= benchmarkTimestamp(payload.generated_at) ? current.king_id : publicationCurrent ? publicationCurrent.king_id || ((publicationCurrent.result || {}).model || {}).king_id : null;
            if (source.results || current) {
                merged.forEach(function(item) { if (item.competition === competition) item.result.model.is_current = item.king_id === currentId; });
            }
        });
        var kings = Array.from(merged.values());
        return { schema_version: "teutonic-king-benchmark-all-results.v2", generated_at: generatedAt, kings: kings,
            benchmark_result_count: kings.reduce(function(count, item) { return count + item.result.benchmarks.filter(function(row) { return row.metric && row.metric.value != null && finiteNumber(row.metric.value, null) != null; }).length; }, 0) };
    }

    function benchmarkPresentation(payload, selectedKingId) {
        if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
            throw new Error("benchmark results must be an object");
        }
        if ([
            "teutonic-king-benchmark-all-results.v1",
            "teutonic-king-benchmark-all-results.v2"
        ].indexOf(payload.schema_version) === -1) {
            throw new Error("unsupported benchmark results schema");
        }
        if (!Array.isArray(payload.kings)) throw new Error("benchmark kings must be an array");
        var kings = payload.kings.map(function(item) {
            item = item || {};
            var result = item.result || {};
            var model = result.model || {};
            var competition = item.competition || model.competition_key || model.competition || "main";
            if (!/^mix0[1-6]$/.test(competition)) competition = "main";
            var byName = {};
            (Array.isArray(result.benchmarks) ? result.benchmarks : []).forEach(function(row) {
                if (row && typeof row.name === "string") byName[benchmarkKey(row.name)] = row;
            });
            var benchmarks = BENCHMARK_SPECS.map(function(spec) {
                var row = byName[benchmarkKey(spec.sourceName || spec.name)] || {};
                var metric = row.metric || {};
                var score = metric.value == null ? null : finiteNumber(metric.value, null);
                return {
                    name: spec.name,
                    status: String(row.status || "pending").toLowerCase(),
                    score: score,
                    metric: typeof metric.name === "string" ? metric.name : null,
                    fewshot: row.fewshot == null ? spec.fewshot : Math.max(0, Math.floor(finiteNumber(row.fewshot, spec.fewshot))),
                    wallTimeSeconds: row.wall_time_s == null ? null : finiteNumber(row.wall_time_s, null)
                };
            });
            var kingId = String(item.king_id || model.king_id || "");
            var scored = benchmarks.filter(function(row) { return row.status === "completed" && row.score != null; });
            return {
                kingId: kingId,
                key: competition + ":" + kingId,
                competition: competition,
                detailsKey: benchmarkDetailsKey(competition, kingId, item.details_key),
                reignNumber: Math.max(0, Math.floor(finiteNumber(model.reign_number, 0))),
                uid: Math.max(0, Math.floor(finiteNumber(model.uid, 0))),
                hotkey: String(model.hotkey || ""),
                modelRepo: String(model.model_repo || "Public king model"),
                crownedAt: model.crowned_at || null,
                timestamp: model.crowned_at || item.updated_at || result.generated_at || null,
                current: model.is_current === true,
                status: String(item.status || result.status || "pending").toLowerCase(),
                updatedAt: item.updated_at || result.generated_at || null,
                completed: benchmarks.filter(function(row) { return row.status === "completed"; }).length,
                scored: scored.length,
                average: scored.length === BENCHMARK_SPECS.length ? scored.reduce(function(sum, row) { return sum + row.score; }, 0) / scored.length : null,
                partialAverage: scored.length ? scored.reduce(function(sum, row) { return sum + row.score; }, 0) / scored.length : null,
                benchmarks: benchmarks
            };
        }).filter(function(king) { return king.kingId; }).sort(function(a, b) {
            return benchmarkTimestamp(b.timestamp) - benchmarkTimestamp(a.timestamp) || b.reignNumber - a.reignNumber || a.competition.localeCompare(b.competition) || a.kingId.localeCompare(b.kingId);
        });
        var ranked = kings.filter(function(king) { return king.average != null; }).sort(function(a, b) { return b.average - a.average; });
        var best = ranked[0] || null;
        var partial = kings.slice().sort(function(a, b) { return b.scored - a.scored || (b.partialAverage == null ? -1 : b.partialAverage) - (a.partialAverage == null ? -1 : a.partialAverage); })[0];
        var selected = kings.find(function(king) { return king.key === selectedKingId || king.kingId === selectedKingId; })
            || best || partial || null;
        var series = BENCHMARK_SPECS.map(function(spec) {
            return {
                name: spec.name,
                points: kings.slice().reverse().map(function(king) {
                    var benchmark = king.benchmarks.find(function(row) { return row.name === spec.name; });
                    return {
                        kingId: king.kingId,
                        competition: king.competition,
                        reignNumber: king.reignNumber,
                        timestamp: king.timestamp,
                        score: benchmark ? benchmark.score : null,
                        status: benchmark ? benchmark.status : "pending",
                        current: king.current
                    };
                }).filter(function(point) { return point.score != null; })
            };
        });
        return {
            generatedAt: payload.generated_at || null,
            kingCount: kings.length,
            benchmarkResultCount: Math.max(0, Math.floor(finiteNumber(payload.benchmark_result_count, 0))),
            kings: kings,
            selected: selected,
            best: best,
            benchmarkCount: BENCHMARK_SPECS.length,
            series: series
        };
    }

    return {
        HIDDEN: HIDDEN,
        validate: validate,
        presentation: presentation,
        currentEvaluationPresentation: currentEvaluationPresentation,
        historyPresentation: historyPresentation,
        verdictLabel: verdictLabel,
        evaluationHistoryMetricsPresentation: evaluationHistoryMetricsPresentation,
        sourceScoresPresentation: sourceScoresPresentation,
        categoryScoresPresentation: categoryScoresPresentation,
        taoMarketCapHotkeyUrl: taoMarketCapHotkeyUrl,
        taoMarketCapColdkeyUrl: taoMarketCapColdkeyUrl,
        shardPresentation: shardPresentation,
        uploadFailurePresentation: uploadFailurePresentation,
        decisionPresentation: decisionPresentation,
        datasetPresentation: datasetPresentation,
        mainDatasetPresentation: mainDatasetPresentation,
        graphPointRadius: graphPointRadius,
        lossChartPresentation: lossChartPresentation,
        datasetChangePresentation: datasetChangePresentation,
        benchmarkDetailsKey: benchmarkDetailsKey,
        benchmarkDetailRequests: benchmarkDetailRequests,
        mergeBenchmarkSources: mergeBenchmarkSources,
        benchmarkPresentation: benchmarkPresentation
    };
});
