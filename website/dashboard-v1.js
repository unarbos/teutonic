(function(root, factory) {
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    root.TeutonicDashboardV1 = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function() {
    "use strict";

    var HIDDEN = "Hidden until promotion";
    var BENCHMARK_SPECS = [
        { name: "BBH", fewshot: 3 },
        { name: "MMLU", fewshot: 5 },
        { name: "HellaSwag", fewshot: 10 },
        { name: "WinoGrande", fewshot: 5 },
        { name: "GSM8K", fewshot: 4 },
        { name: "PIQA", fewshot: 0 },
        { name: "ARC-C", fewshot: 25 },
        { name: "ARC-E", fewshot: 0 },
        { name: "GPQA Diamond", fewshot: 5 },
        { name: "MATH-500", fewshot: 4 }
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
            var names = [];
            rawNames.forEach(function(value) {
                if (typeof value !== "string" || !value.trim()) return;
                var name = value.trim();
                if (names.indexOf(name) === -1) names.push(name);
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
            var byName = {};
            (Array.isArray(result.benchmarks) ? result.benchmarks : []).forEach(function(row) {
                if (row && typeof row.name === "string") byName[benchmarkKey(row.name)] = row;
            });
            var benchmarks = BENCHMARK_SPECS.map(function(spec) {
                var row = byName[benchmarkKey(spec.name)] || {};
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
            return {
                kingId: String(item.king_id || model.king_id || ""),
                reignNumber: Math.max(0, Math.floor(finiteNumber(model.reign_number, 0))),
                uid: Math.max(0, Math.floor(finiteNumber(model.uid, 0))),
                hotkey: String(model.hotkey || ""),
                modelRepo: String(model.model_repo || "Public king model"),
                crownedAt: model.crowned_at || null,
                current: model.is_current === true,
                status: String(item.status || result.status || "pending").toLowerCase(),
                updatedAt: item.updated_at || result.generated_at || null,
                completed: benchmarks.filter(function(row) { return row.status === "completed"; }).length,
                benchmarks: benchmarks
            };
        }).filter(function(king) { return king.kingId; }).sort(function(a, b) {
            return b.reignNumber - a.reignNumber || String(b.crownedAt || "").localeCompare(String(a.crownedAt || ""));
        });
        var selected = kings.find(function(king) { return king.kingId === selectedKingId; })
            || kings.find(function(king) { return king.current; }) || kings[0] || null;
        var series = BENCHMARK_SPECS.map(function(spec) {
            return {
                name: spec.name,
                points: kings.slice().reverse().map(function(king) {
                    var benchmark = king.benchmarks.find(function(row) { return row.name === spec.name; });
                    return {
                        kingId: king.kingId,
                        reignNumber: king.reignNumber,
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
        taoMarketCapHotkeyUrl: taoMarketCapHotkeyUrl,
        taoMarketCapColdkeyUrl: taoMarketCapColdkeyUrl,
        shardPresentation: shardPresentation,
        uploadFailurePresentation: uploadFailurePresentation,
        decisionPresentation: decisionPresentation,
        datasetPresentation: datasetPresentation,
        graphPointRadius: graphPointRadius,
        lossChartPresentation: lossChartPresentation,
        datasetChangePresentation: datasetChangePresentation,
        benchmarkPresentation: benchmarkPresentation
    };
});
