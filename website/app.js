(function () {
  "use strict";
  var ENDPOINT = "/dashboard.json";
  var DATASET_MANIFEST_URL = "https://pub-fedac496355c4edc9aed57189e6e190f.r2.dev/datasets/manifest.json";
  var BENCHMARK_RESULTS_URL = "https://pub-c982d552b8044578b4a79e653700ec73.r2.dev/king-benchmark-daily/all-kings/results.json";
  var MODEL_STORAGE_BASE = "https://pub-0821d4e196224864af220294345fd141.r2.dev/";
  var POLL_MS = 15000;
  var DATASET_POLL_MS = 60000;
  var BENCHMARK_POLL_MS = 60000;
  var lastPayload = null;
  var mainQueueFailed = false;
  var benchmarkPayload = null;
  var benchmarkHistoryVisible = false;
  var historyShowErrors = false;
  var historyCompetition = "all";
  var historyExpandedDetails = new Set();
  var smoothMode = localStorage.getItem("smoothMode") || "lowess";
  if (smoothMode !== "lowess" && smoothMode !== "normal") smoothMode = "lowess";
  function el(id) { return document.getElementById(id); }
  function text(id, value) { el(id).textContent = value == null || value === "" ? "--" : String(value); }
  function finite(value) { if (value == null || value === "") return null; var n = Number(value); return Number.isFinite(n) ? n : null; }
  function number(value, digits) { var n = finite(value); return n == null ? "--" : n.toLocaleString(undefined, { minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0 }); }
  function metric(value, digits) { var n = finite(value); return n == null ? "--" : n.toFixed(digits == null ? 6 : digits); }
  function percent(value) { var n = finite(value); return n == null ? "--" : (n * 100).toFixed(1) + "%"; }
  function usd(value) { var n = finite(value); return n == null ? "--" : "$" + n.toFixed(2); }
  function short(value, head, tail) { var s = String(value || ""); head = head || 9; tail = tail || 5; return s.length > head + tail + 1 ? s.slice(0, head) + "…" + s.slice(-tail) : (s || "--"); }
  function date(value) { if (!value) return "--"; var d = new Date(value); return Number.isNaN(d.getTime()) ? "--" : d.toLocaleString([], { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }); }
  function age(value) { if (!value) return "--"; var seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000)); if (seconds < 60) return seconds + "S AGO"; if (seconds < 3600) return Math.floor(seconds / 60) + "M AGO"; if (seconds < 86400) return Math.floor(seconds / 3600) + "H AGO"; return Math.floor(seconds / 86400) + "D AGO"; }
  function identity(row) { if (!row) return "--"; if (row.model_identity === "hidden_until_promotion") return TeutonicDashboardV1.HIDDEN; return row.challenger_repo || row.model_repo || (row.model_digest ? "SHA256 " + short(row.model_digest, 10, 4) : "PUBLIC MODEL"); }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function cell(row, value, className, title) { var td = document.createElement("td"); td.textContent = value == null ? "--" : value; if (className) td.className = className; if (title) td.title = title; row.appendChild(td); return td; }
  function hotkeyLink(hotkey, head, tail) { var value = String(hotkey || ""), href = TeutonicDashboardV1.taoMarketCapHotkeyUrl(value), node = document.createElement(href ? "a" : "span"); node.textContent = short(value, head, tail); node.title = value ? "Open " + value + " on Tao Market Cap" : ""; if (href) { node.href = href; node.target = "_blank"; node.rel = "noopener"; } return node; }
  function hotkeyCell(row, hotkey) { var td = cell(row, "", "mono", hotkey); td.appendChild(hotkeyLink(hotkey, 4, 4)); return td; }
  function coldkeyCell(row, coldkey) { var value = String(coldkey || ""), href = TeutonicDashboardV1.taoMarketCapColdkeyUrl(value), td = cell(row, "", "mono", value), node = document.createElement(href ? "a" : "span"); node.textContent = short(value, 4, 4); node.title = value ? "Open " + value + " on Tao Market Cap" : ""; if (href) { node.href = href; node.target = "_blank"; node.rel = "noopener"; } td.appendChild(node); return td; }
  function hotkeyText(id, prefix, hotkey, head, tail) { var node = el(id); clear(node); node.title = hotkey || ""; node.appendChild(document.createTextNode(prefix)); node.appendChild(hotkeyLink(hotkey, head, tail)); }
  function emptyRow(body, columns, message) { clear(body); var row = document.createElement("tr"); var td = cell(row, message, "empty-cell"); td.colSpan = columns; body.appendChild(row); }
  function historyDetailKey(item, index) { return String(item.challenge_id || item.upload_id || item.timestamp || "evaluation-" + index); }
  function historySourceScores(item) {
    var view = TeutonicDashboardV1.sourceScoresPresentation(item), section = document.createElement("section"), heading = document.createElement("h3"), wrap = document.createElement("div"), table = document.createElement("table"), head = document.createElement("thead"), headRow = document.createElement("tr"), body = document.createElement("tbody");
    section.className = "history-source-scores"; heading.textContent = "LOSS BY DATASET · " + view.count; section.appendChild(heading);
    if (!view.count) { var empty = document.createElement("p"); empty.className = "history-shards-empty"; empty.textContent = "PER-DATASET LOSS UNAVAILABLE FOR THIS EVALUATION"; section.appendChild(empty); return section; }
    wrap.className = "history-source-score-wrap"; table.className = "history-source-score-table";
    ["DATASET", "KING LOSS", "CHALL LOSS", "Δ", "N"].forEach(function(label) { var th = document.createElement("th"); th.textContent = label; headRow.appendChild(th); });
    head.appendChild(headRow); table.appendChild(head);
    view.rows.forEach(function(score) { var row = document.createElement("tr"), delta = finite(score.muHat); cell(row, score.source, "history-source-name", score.source); cell(row, metric(score.kingLoss, 5), "mono"); cell(row, metric(score.challengerLoss, 5), "mono"); cell(row, delta == null ? "--" : (delta > 0 ? "+" : "") + delta.toFixed(5), "mono history-source-delta " + (delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral")); cell(row, number(score.nSequences), "mono"); body.appendChild(row); });
    table.appendChild(body); wrap.appendChild(table); section.appendChild(wrap); return section;
  }
  function historyShardRow(item, index, detailKey) {
    var view = TeutonicDashboardV1.shardPresentation(item), uploadFailure = TeutonicDashboardV1.uploadFailurePresentation(item), run = TeutonicDashboardV1.evaluationHistoryMetricsPresentation(item), decision = TeutonicDashboardV1.decisionPresentation(item), row = document.createElement("tr"), td = cell(row, "", "history-shards-cell"), panel = document.createElement("div"), reason = document.createElement("div"), reasonLabel = document.createElement("strong"), reasonCopy = document.createElement("div"), reasonSummary = document.createElement("p"), reasonDetail = document.createElement("span"), heading = document.createElement("strong"), detailType = uploadFailure ? "Upload failure" : "Evaluation";
    row.className = "history-shards-row"; row.id = "history-shards-" + detailKey.replace(/[^a-zA-Z0-9_-]/g, "") + "-" + index; row.hidden = !historyExpandedDetails.has(detailKey); row.setAttribute("role", "region"); row.setAttribute("aria-label", detailType + " details for " + (item.challenge_id || item.upload_id || index + 1)); td.colSpan = 11; panel.className = "history-shards-panel";
    reason.className = "history-decision " + decision.kind; reasonLabel.textContent = decision.label; reasonSummary.textContent = decision.summary; reasonDetail.textContent = decision.detail; reasonCopy.appendChild(reasonSummary); if (decision.detail) reasonCopy.appendChild(reasonDetail); reason.appendChild(reasonLabel); reason.appendChild(reasonCopy); panel.appendChild(reason);
    if (uploadFailure) {
      var metadata = document.createElement("dl"); metadata.className = "history-error-metadata";
      [["REGISTRATION", uploadFailure.registration], ["UPLOAD", uploadFailure.uploadId], ["UPLOAD STATE", uploadFailure.uploadState], ["FAILURE CODE", uploadFailure.failureCode]].forEach(function (entry) { var group = document.createElement("div"), label = document.createElement("dt"), value = document.createElement("dd"); label.textContent = entry[0]; value.textContent = entry[1]; group.appendChild(label); group.appendChild(value); metadata.appendChild(group); });
      panel.appendChild(metadata); td.appendChild(panel); return row;
    }
    var evaluationMetadata = document.createElement("dl"); evaluationMetadata.className = "history-error-metadata history-eval-metadata";
    [["SAMPLES", run.samples, run.samplesTitle], ["EARLY STOP", run.earlyStopLabel, "Whether evaluation stopped before all planned samples"]].forEach(function(entry) { var group = document.createElement("div"), label = document.createElement("dt"), value = document.createElement("dd"); label.textContent = entry[0]; value.textContent = entry[1]; value.title = entry[2]; group.appendChild(label); group.appendChild(value); evaluationMetadata.appendChild(group); });
    panel.appendChild(evaluationMetadata);
    panel.appendChild(historySourceScores(item));
    heading.className = "history-shards-heading"; heading.textContent = "SHARDS USED · " + view.count; panel.appendChild(heading);
    if (!view.count) { var empty = document.createElement("p"); empty.className = "history-shards-empty"; empty.textContent = "SHARD DATA UNAVAILABLE FOR THIS EVALUATION"; panel.appendChild(empty); }
    view.groups.forEach(function (group) { var section = document.createElement("section"), label = document.createElement("h3"), list = document.createElement("ul"); section.className = "history-shard-group"; label.textContent = group.source + " · " + group.names.length; group.names.forEach(function (name) { var entry = document.createElement("li"), code = document.createElement("code"); code.textContent = name; entry.appendChild(code); list.appendChild(entry); }); section.appendChild(label); section.appendChild(list); panel.appendChild(section); });
    td.appendChild(panel); return row;
  }
  function makeHistoryRowExpandable(row, details, item, detailKey) {
    var detailType = item.upload_id ? "upload failure" : "evaluation", labelId = item.challenge_id || item.upload_id || "", initiallyExpanded = historyExpandedDetails.has(detailKey); row.classList.add("history-row"); row.tabIndex = 0; row.setAttribute("aria-expanded", initiallyExpanded ? "true" : "false"); row.setAttribute("aria-controls", details.id); row.setAttribute("aria-label", (initiallyExpanded ? "Hide" : "Show") + " details for " + detailType + " " + labelId); row.title = "Click to " + (initiallyExpanded ? "hide" : "show") + " " + detailType + " details";
    function toggle(event) { if (event.type === "click" && event.target.closest("a,button")) return; if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return; if (event.type === "keydown") event.preventDefault(); var expanded = row.getAttribute("aria-expanded") === "true"; if (expanded) historyExpandedDetails.delete(detailKey); else historyExpandedDetails.add(detailKey); row.setAttribute("aria-expanded", expanded ? "false" : "true"); row.setAttribute("aria-label", (expanded ? "Show" : "Hide") + " details for " + detailType + " " + labelId); row.title = expanded ? "Click to show " + detailType + " details" : "Click to hide " + detailType + " details"; details.hidden = expanded; }
    row.addEventListener("click", toggle); row.addEventListener("keydown", toggle);
  }
  function ema(values, alpha) { if (alpha >= 1 || !values.length) return values.slice(); var output = [values[0]]; for (var i = 1; i < values.length; i++) output.push(alpha * values[i] + (1 - alpha) * output[i - 1]); return output; }
  function lowess(values, strength) { var n = values.length; if (n < 3 || strength <= 0) return values.slice(); var span = Math.min(n, Math.max(3, Math.ceil(n * (.12 + strength * .58)))), output = []; for (var i = 0; i < n; i++) { var distances = []; for (var j = 0; j < n; j++) distances.push({ index: j, distance: Math.abs(j - i) }); distances.sort(function (a, b) { return a.distance - b.distance; }); var bandwidth = distances[span - 1].distance || 1, sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0; for (var k = 0; k < span; k++) { var index = distances[k].index, u = distances[k].distance / bandwidth, weight = Math.pow(1 - Math.pow(u, 3), 3), xValue = index, yValue = values[index]; sw += weight; swx += weight * xValue; swy += weight * yValue; swxx += weight * xValue * xValue; swxy += weight * xValue * yValue; } var denominator = sw * swxx - swx * swx; output.push(Math.abs(denominator) < 1e-12 || sw === 0 ? (sw ? swy / sw : values[i]) : (swy - ((sw * swxy - swx * swy) / denominator) * swx) / sw + ((sw * swxy - swx * swy) / denominator) * i); } return output; }
  function smoothSeries(values, amount) { if (amount <= 0) return values.slice(); return smoothMode === "normal" ? ema(values, 1 / (1 + amount * 20)) : lowess(values, amount); }

  function compactNumber(value) {
    var n = finite(value); if (n == null) return "--";
    var abs = Math.abs(n), units = [{ value: 1e12, suffix: "T" }, { value: 1e9, suffix: "B" }, { value: 1e6, suffix: "M" }, { value: 1e3, suffix: "K" }];
    for (var i = 0; i < units.length; i++) { if (abs >= units[i].value) { var scaled = n / units[i].value; return scaled.toFixed(Math.abs(scaled) >= 100 ? 0 : 2).replace(/\.00$/, "") + units[i].suffix; } }
    return number(n);
  }
  function datasetWeight(value) { var n = finite(value); if (n == null) return "--"; var pct = n * 100; return (Math.abs(pct - Math.round(pct)) < .05 ? String(Math.round(pct)) : pct.toFixed(1)) + "%"; }
  function smallPercent(value) { var n = finite(value); if (n == null) return "--"; if (n === 0) return "0%"; if (Math.abs(n) < .01) return n.toPrecision(2) + "%"; return (Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(3)).replace(/0+$/, "").replace(/\.$/, "") + "%"; }
  function revision(value) { var parts = String(value || "").split(":"); return parts.length === 2 ? parts[0] + ":" + parts[1].slice(0, 16) : short(value, 16, 0); }
  function compactTimestamp(value) { if (!value) return "--"; var d = new Date(value); return Number.isNaN(d.getTime()) ? "--" : d.toISOString().replace(/:\d{2}\.\d{3}Z$/, "Z"); }
  function setLink(id, label, href) { var node = el(id); node.textContent = label || "--"; node.href = href || "#"; if (href) { node.target = "_blank"; node.rel = "noopener"; } else { node.removeAttribute("target"); node.removeAttribute("rel"); } }
  function huggingFaceUrl(repo, digest) { var base = repo ? "https://huggingface.co/" + repo : ""; var raw = String(digest || "").replace(/^hf:/, ""); return base && raw ? base + "/tree/" + raw : base; }
  function datasetCell(row, value, subtext, href) {
    var td = document.createElement("td"), main;
    if (href) { main = document.createElement("a"); main.href = href; main.target = "_blank"; main.rel = "noopener"; main.textContent = value || "--"; td.appendChild(main); }
    else td.textContent = value || "--";
    if (subtext) { var sub = document.createElement("span"); sub.className = "dataset-sub"; sub.textContent = subtext; td.appendChild(sub); }
    row.appendChild(td);
  }
  async function fetchJson(url) {
    var separator = url.indexOf("?") === -1 ? "?" : "&";
    var response = await fetch(url + separator + "t=" + Date.now(), { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    return response.json();
  }
  async function fetchFirstJson(urls) {
    var lastError;
    for (var i = 0; i < urls.length; i++) { try { return await fetchJson(urls[i]); } catch (error) { lastError = error; } }
    throw lastError || new Error("no JSON endpoint available");
  }
  function benchmarkSparkline(series) {
    var namespace = "http://www.w3.org/2000/svg", svg = document.createElementNS(namespace, "svg"), points = series.points || [], W = 220, H = 50, left = 8, right = 8, top = 5, bottom = 14;
    function node(name, attributes) { var item = document.createElementNS(namespace, name); Object.keys(attributes || {}).forEach(function(key) { item.setAttribute(key, attributes[key]); }); return item; }
    svg.classList.add("bench-sparkline"); svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", series.name + " score history across " + points.length + (points.length === 1 ? " reign" : " reigns"));
    svg.appendChild(node("line", { x1: left, y1: H - bottom, x2: W - right, y2: H - bottom, class: "bench-sparkline-axis" }));
    if (!points.length) { var empty = node("text", { x: W / 2, y: H / 2, class: "bench-sparkline-empty", "text-anchor": "middle" }); empty.textContent = "NO SCORES"; svg.appendChild(empty); return svg; }
    var values = points.map(function(point) { return point.score; }), min = Math.min.apply(null, values), max = Math.max.apply(null, values), spread = max - min, padding = spread ? spread * .18 : Math.max(Math.abs(max) * .05, .01); min -= padding; max += padding;
    function x(index) { return points.length === 1 ? W / 2 : left + index / (points.length - 1) * (W - left - right); }
    function y(value) { return top + (max - value) / (max - min) * (H - top - bottom); }
    if (points.length > 1) svg.appendChild(node("polyline", { points: points.map(function(point, index) { return x(index).toFixed(1) + "," + y(point.score).toFixed(1); }).join(" "), class: "bench-sparkline-line" }));
    var pointRadius = TeutonicDashboardV1.graphPointRadius(points.length, W - left - right, 1, 2.7);
    points.forEach(function(point, index) { var radius = point.current ? Math.min(3.5, pointRadius + 0.8) : pointRadius, dot = node("circle", { cx: x(index).toFixed(1), cy: y(point.score).toFixed(1), r: radius.toFixed(2), class: point.current ? "bench-sparkline-point current" : "bench-sparkline-point" }), title = node("title"); title.textContent = "REIGN #" + point.reignNumber + " · " + percent(point.score); dot.appendChild(title); svg.appendChild(dot); });
    var first = node("text", { x: left, y: H - 3, class: "bench-sparkline-label", "text-anchor": "start" }); first.textContent = "R" + points[0].reignNumber; svg.appendChild(first);
    if (points.length > 1) { var last = node("text", { x: W - right, y: H - 3, class: "bench-sparkline-label", "text-anchor": "end" }); last.textContent = "R" + points[points.length - 1].reignNumber; svg.appendChild(last); }
    return svg;
  }
  function renderBenchmarkHistory(view) {
    var panel = el("benchmark-history"), graphs = el("benchmark-history-graphs"), body = el("benchmark-history-body"), toggle = el("benchmark-history-toggle");
    panel.hidden = !benchmarkHistoryVisible; toggle.textContent = benchmarkHistoryVisible ? "HIDE HISTORY" : "SHOW HISTORY"; toggle.setAttribute("aria-pressed", benchmarkHistoryVisible ? "true" : "false");
    clear(graphs); clear(body); if (!benchmarkHistoryVisible) return;
    view.series.forEach(function(series) { var tile = document.createElement("article"), heading = document.createElement("div"), name = document.createElement("strong"), latest = document.createElement("span"), lastPoint = series.points.length ? series.points[series.points.length - 1] : null; tile.className = "bench-history-graph"; heading.className = "bench-history-graph-head"; name.textContent = series.name; latest.textContent = lastPoint ? "LATEST " + percent(lastPoint.score) : "NO SCORES"; heading.appendChild(name); heading.appendChild(latest); tile.appendChild(heading); tile.appendChild(benchmarkSparkline(series)); graphs.appendChild(tile); });
    if (!view.kings.length) return emptyRow(body, view.series.length + 3, "NO BENCHMARK HISTORY YET");
    view.kings.forEach(function(king) { var row = document.createElement("tr"), byName = {}; if (king.current) row.className = "bench-history-current"; king.benchmarks.forEach(function(benchmark) { byName[benchmark.name] = benchmark; }); cell(row, "#" + king.reignNumber); cell(row, king.uid); cell(row, king.modelRepo, "", king.hotkey); view.series.forEach(function(series) { var benchmark = byName[series.name]; cell(row, benchmark && benchmark.score != null ? percent(benchmark.score) : "--", "mono", benchmark ? benchmark.status.toUpperCase() : "PENDING"); }); body.appendChild(row); });
  }
  function renderBenchmarks(payload) {
    var view = TeutonicDashboardV1.benchmarkPresentation(payload), selected = view.selected, meta = el("benchmark-meta"), grid = el("benchmark-grid");
    benchmarkPayload = payload;
    meta.textContent = view.kingCount + (view.kingCount === 1 ? " KING" : " KINGS") + " · " + view.benchmarkResultCount + " RESULTS · " + age(view.generatedAt);
    renderBenchmarkHistory(view);
    if (!selected) { text("benchmark-model", "NO KING BENCHMARK RESULTS YET"); text("benchmark-identity", "THE DAILY BENCHMARK QUEUE HAS NOT PUBLISHED A RESULT"); clear(grid); var none = document.createElement("p"); none.className = "bench-empty"; none.textContent = "WAITING FOR THE FIRST KING RESULT"; grid.appendChild(none); return; }
    text("benchmark-model", selected.modelRepo);
    var identityNode = el("benchmark-identity"); clear(identityNode); identityNode.appendChild(document.createTextNode("REIGN #" + selected.reignNumber + " · UID " + selected.uid + " · ")); identityNode.appendChild(hotkeyLink(selected.hotkey, 12, 6)); identityNode.appendChild(document.createTextNode(" · " + selected.completed + "/" + view.benchmarkCount + " COMPLETE · UPDATED " + age(selected.updatedAt)));
    clear(grid);
    selected.benchmarks.forEach(function (benchmark) { var card = document.createElement("article"), name = document.createElement("div"), score = document.createElement("div"), status = document.createElement("div"), state = document.createElement("span"), shots = document.createElement("span"); card.className = "bench-card"; card.dataset.status = benchmark.status; name.className = "bench-name"; name.textContent = benchmark.name; score.className = "bench-score"; score.textContent = benchmark.score == null ? "--" : percent(benchmark.score); score.title = [benchmark.metric, finite(benchmark.wallTimeSeconds) == null ? "" : metric(benchmark.wallTimeSeconds, 0) + " seconds"].filter(Boolean).join(" · "); status.className = "bench-status"; state.textContent = benchmark.status.replaceAll("_", " "); shots.textContent = benchmark.fewshot + "-SHOT"; status.appendChild(state); status.appendChild(shots); card.appendChild(name); card.appendChild(score); card.appendChild(status); grid.appendChild(card); });
  }
  async function loadBenchmarks() {
    try { renderBenchmarks(await fetchFirstJson(["/benchmarks/results.json", BENCHMARK_RESULTS_URL])); }
    catch (error) { if (!benchmarkPayload) { text("benchmark-meta", "BENCHMARK RESULTS UNAVAILABLE"); clear(el("benchmark-grid")); var empty = document.createElement("p"); empty.className = "bench-empty"; empty.textContent = "LIVE BENCHMARK RESULTS COULD NOT BE LOADED"; el("benchmark-grid").appendChild(empty); } }
  }
  function renderDatasetManifest(manifest) {
    var view = TeutonicDashboardV1.datasetPresentation(manifest), summary = [view.rows.length + (view.rows.length === 1 ? " DATASET" : " DATASETS")];
    if (view.totalTokens) summary.push(compactNumber(view.totalTokens) + " TOKENS");
    if (view.sequenceLength) summary.push("SEQ LEN " + number(view.sequenceLength));
    if (view.evalN) summary.push("EVAL SAMPLE " + number(view.evalN) + " SEQ" + (view.evalTokens ? " / " + compactNumber(view.evalTokens) + " TOKENS" : ""));
    summary.push("PLUS ~6M DOCUMENT TOKENS (" + number(2049) + "–" + number(8192) + " TOKENS/DOC)");
    text("dataset-summary", summary.join(" · "));
    if (view.tokenizer) {
      var tokenizer = document.createElement("span"); tokenizer.className = "dataset-tokenizer"; tokenizer.textContent = "TOKENIZER: " + view.tokenizer;
      el("dataset-summary").appendChild(tokenizer);
    }
    var body = el("dataset-sources"); clear(body);
    if (!view.rows.length) return emptyRow(body, 6, "NO DATASETS IN MANIFEST");
    view.rows.forEach(function (source) {
      var tr = document.createElement("tr");
      datasetCell(tr, source.name, source.metadataLoaded ? source.tokenizationMode : "METADATA UNAVAILABLE", source.manifestUrl);
      datasetCell(tr, datasetWeight(source.weight), "NORMALIZED " + datasetWeight(source.normalizedWeight));
      datasetCell(tr, source.totalTokens ? compactNumber(source.totalTokens) : "--", source.totalShards ? number(source.totalShards) + " SHARDS" : "");
      datasetCell(tr, source.sequences ? compactNumber(source.sequences) : "--", source.sequenceLength ? "LEN " + number(source.sequenceLength) : "");
      datasetCell(tr, source.evalSequences ? number(source.evalSequences) + " SEQ" : "--", source.evalTokens ? compactNumber(source.evalTokens) + " TOKENS · " + smallPercent(source.sampleRate) + " SAMPLED" : "");
      datasetCell(tr, source.source || "--", [source.tokenizer, source.dtype].filter(Boolean).join(" · "));
      body.appendChild(tr);
    });
  }
  async function loadDatasetManifest() {
    el("dataset-manifest-link").href = DATASET_MANIFEST_URL;
    try {
      var manifest = await fetchFirstJson(["/datasets/manifest.json", DATASET_MANIFEST_URL]);
      renderDatasetManifest(manifest);
    } catch (error) {
      text("dataset-summary", "MANIFEST UNAVAILABLE · USE THE MANIFEST LINK");
      emptyRow(el("dataset-sources"), 6, "DATASET MANIFEST COULD NOT BE LOADED");
    }
  }

  function renderHeader(d) {
    var chain = d.chain || {}, king = d.king || {}, market = d.market || {};
    text("tao-price", finite(market.tao_price_usd) == null ? "--" : usd(market.tao_price_usd));
    var change = finite(market.tao_change_24h), changeNode = el("tao-change");
    changeNode.textContent = change == null ? "" : (change > 0 ? "+" : "") + change.toFixed(1) + "%";
    changeNode.className = change == null ? "" : (change >= 0 ? "is-up" : "is-down");
    text("sn3-alpha", finite(market.sn3_alpha_price_tao) == null ? "--" : metric(market.sn3_alpha_price_tao, 4) + " τ");
    text("sn3-reg", finite(market.sn3_reg_burn_tao) == null ? "--" : metric(market.sn3_reg_burn_tao, 6) + " τ");

    var genesisRepo = chain.seed_repo, genesisDigest = chain.seed_digest;
    var genesisUrl = chain.seed_repo_backend === "hf" ? huggingFaceUrl(genesisRepo, genesisDigest) : "";
    setLink("genesis-link", genesisRepo, genesisUrl);
    setLink("genesis-revision", revision(genesisDigest), genesisUrl);

    var genesisKing = Number(king.reign_number) === 0 || !king.model_repo;
    var kingRepo = genesisKing ? genesisRepo : king.model_repo;
    var kingDigest = genesisKing ? genesisDigest : (king.king_digest || king.model_digest);
    var kingUrl = genesisKing ? genesisUrl : (king.model_reference ? new URL(king.model_reference + "manifest.json", MODEL_STORAGE_BASE).href : "");
    setLink("king-link", d.king ? kingRepo : "Awaiting first winner", d.king ? kingUrl : "");
    setLink("king-revision", d.king ? revision(kingDigest) : "--", d.king ? kingUrl : "");
    el("king-health").classList.toggle("is-live", !!d.king);
    text("king-reign", "REIGN " + (d.king ? "#" + number(king.reign_number) + " — " + compactTimestamp(king.crowned_at) : "--"));
    text("source-watermark", "WATERMARK " + number(d.source_watermark));
    document.title = (chain.name || "Teutonic") + " — Dashboard";
  }
  function competitionSources() {
    return [{ key: "main", dashboard: lastPayload, failed: mainQueueFailed }].concat(specialists || []);
  }
  function globalQueueRows() {
    var rows = [];
    competitionSources().forEach(function (source) {
      if (source.dashboard) source.dashboard.queue.forEach(function (item) { rows.push({ competition: source.key, item: item }); });
    });
    return rows.sort(function (a, b) {
      var aPosition = finite(a.item.queue_position), bPosition = finite(b.item.queue_position);
      return (aPosition == null ? Infinity : aPosition) - (bPosition == null ? Infinity : bPosition)
        || Number(a.item.block || 0) - Number(b.item.block || 0)
        || String(a.item.submitted_at || "").localeCompare(String(b.item.submitted_at || ""))
        || String(a.item.challenge_id || a.item.hotkey || "").localeCompare(String(b.item.challenge_id || b.item.hotkey || ""));
    });
  }
  function renderEvaluation() {
    var sources = competitionSources(), missing = [], stale = [];
    sources.forEach(function (source) { if (!source.dashboard) missing.push(source.key.toUpperCase()); else if (source.failed) stale.push(source.key.toUpperCase()); });
    var notes = [];
    if (missing.length) notes.push("WAITING FOR EVALUATION DATA: " + missing.join(", "));
    if (stale.length) notes.push("LAST KNOWN EVALUATION DATA: " + stale.join(", "));
    text("eval-data-status", notes.join(" · ")); el("eval-data-status").hidden = !notes.length;
    // Publications arrive independently; prefer the freshest active snapshot.
    var active = sources.filter(function (source) { return source.dashboard && source.dashboard.current_eval; }).sort(function (a, b) {
      return Number(a.failed) - Number(b.failed)
        || new Date(b.dashboard.updated_at || b.dashboard.generated_at || 0) - new Date(a.dashboard.updated_at || a.dashboard.generated_at || 0);
    });
    var selected = active[0], d = selected ? selected.dashboard : lastPayload || {}, ev = d.current_eval;
    var service = d.service_status || {}, card = el("eval-card"), provisionalNode = el("eval-provisional");
    text("validator-phase", "VALIDATOR " + String(service.validator_phase || "--").toUpperCase());
    if (!ev) {
      var next = globalQueueRows()[0];
      card.dataset.active = "false"; provisionalNode.hidden = true;
      text("eval-title", notes.length ? "EVALUATION STATUS INCOMPLETE" : next ? "NEXT CHALLENGE QUEUED · " + next.competition.toUpperCase() : "NO ACTIVE CHALLENGE");
      el("eval-title").title = next ? next.item.hotkey || "" : "";
      if (next) hotkeyText("eval-meta", next.competition.toUpperCase() + " · NEXT HOTKEY ", next.item.hotkey, 16, 8);
      else text("eval-meta", notes.length ? "WAITING FOR ALL COMPETITION PUBLICATIONS" : "THE VALIDATOR IS READY FOR THE NEXT MODEL");
      text("eval-stage", "WAITING"); text("eval-percent", "0%"); el("eval-progress").style.width = "0%"; return;
    }
    var pct = finite(ev.percent); if (pct == null && finite(ev.total) > 0) pct = finite(ev.progress) / finite(ev.total) * 100; pct = Math.max(0, Math.min(100, pct || 0));
    var provisional = TeutonicDashboardV1.currentEvaluationPresentation(ev);
    provisionalNode.hidden = false; text("eval-mu-hat", metric(provisional.muHat)); text("eval-lcb", metric(provisional.lcb)); text("eval-delta", metric(provisional.threshold));
    text("eval-lcb-meta", provisional.available ? number(provisional.sequences) + " PAIRED · " + number(provisional.bootstraps) + " BOOTSTRAPS · UPDATES AT 10% CHECKPOINTS" : "WAITING FOR FIRST 10% CHECKPOINT");
    card.dataset.active = "true";
    hotkeyText("eval-title", selected.key.toUpperCase() + " · HOTKEY ", ev.hotkey, 16, 8); el("eval-title").title = ev.hotkey || "";
    text("eval-meta", "UID " + ev.uid + " · MODEL SHA256 " + short(ev.model_digest, 12, 6) + " · " + number(ev.elapsed_seconds) + "S ELAPSED");
    el("eval-meta").title = ev.model_digest || ""; text("eval-stage", String(ev.stage || "PROCESSING").replaceAll("_", " "));
    text("eval-percent", pct.toFixed(0) + "% · " + number(ev.progress) + "/" + number(ev.total)); el("eval-progress").style.width = pct + "%";
  }
  function renderSharedEvaluation() { renderQueue(); renderEvaluation(); }
  function renderQueue() {
    var sources = competitionSources(), rows = globalQueueRows(), missing = [], stale = [];
    sources.forEach(function (source) {
      if (!source.dashboard) missing.push(source.key.toUpperCase());
      else if (source.failed) stale.push(source.key.toUpperCase());
    });
    text("queue-count", rows.length + (rows.length === 1 ? " MODEL" : " MODELS"));
    var status = [];
    if (missing.length) status.push("WAITING FOR QUEUE DATA: " + missing.join(", "));
    if (stale.length) status.push("LAST KNOWN QUEUE DATA: " + stale.join(", "));
    text("queue-status", status.length ? status.join(" · ") : "MAIN · MATH · CODE · TEXT — GLOBAL READY-COMMIT ORDER");
    var body = el("queue-body"); clear(body);
    if (!rows.length) return emptyRow(body, 9, missing.length || stale.length ? "QUEUE DATA INCOMPLETE" : "QUEUE EMPTY");
    rows.forEach(function (entry) {
      var item = entry.item, row = document.createElement("tr");
      cell(row, item.queue_position == null ? "--" : "#" + item.queue_position);
      cell(row, entry.competition.toUpperCase()); cell(row, item.uid);
      cell(row, item.model_digest, "mono", item.model_digest); hotkeyCell(row, item.hotkey); coldkeyCell(row, item.coldkey);
      cell(row, number(item.block)); cell(row, String(item.state || "queued").toUpperCase()); cell(row, date(item.submitted_at));
      body.appendChild(row);
    });
  }
  function renderCompetitionStatus(id, competition) {
    var notes = [], missing = [], stale = [];
    competitionSources().forEach(function (source) {
      if (competition && competition !== "all" && source.key !== competition) return;
      if (!source.dashboard) missing.push(source.key.toUpperCase());
      else if (source.failed) stale.push(source.key.toUpperCase());
    });
    if (missing.length) notes.push("WAITING FOR DATA: " + missing.join(", "));
    if (stale.length) notes.push("SHOWING LAST PUBLICATION: " + stale.join(", "));
    text(id, notes.join(" · ")); el(id).hidden = !notes.length;
  }
  function renderHistory() {
    var history = [];
    competitionSources().forEach(function (source) {
      if (historyCompetition !== "all" && source.key !== historyCompetition) return;
      if (source.dashboard) (source.dashboard.history || []).forEach(function (item) {
        history.push(Object.assign({}, item, { competition: source.key }));
      });
    });
    renderCompetitionStatus("history-status", historyCompetition);
    var body = el("history-body"), view = TeutonicDashboardV1.historyPresentation(history, historyShowErrors), rows = view.rows.slice().sort(function (a, b) { return new Date(b.timestamp || 0) - new Date(a.timestamp || 0); });
    var countLabel = rows.length + (rows.length === 1 ? " RESULT" : " RESULTS");
    if (!historyShowErrors && view.errorCount) countLabel += " · " + view.errorCount + (view.errorCount === 1 ? " ERROR HIDDEN" : " ERRORS HIDDEN");
    text("history-count", countLabel);
    var toggle = el("history-errors-toggle"); toggle.textContent = historyShowErrors ? "HIDE ERRORS" : "SHOW ERRORS"; toggle.setAttribute("aria-pressed", historyShowErrors ? "true" : "false");
    if (!rows.length) return emptyRow(body, 11, view.errorCount && !historyShowErrors ? "NO NON-ERROR EVALUATIONS — ERRORS HIDDEN" : "NO EVALUATIONS YET"); clear(body);
    rows.forEach(function (item, index) { var detailKey = item.competition + "-" + historyDetailKey(item, index), tr = document.createElement("tr"), details = historyShardRow(item, index, detailKey); cell(tr, item.competition.toUpperCase()); cell(tr, item.uid); cell(tr, identity(item), "", item.challenger_repo || item.challenge_id); hotkeyCell(tr, item.hotkey); coldkeyCell(tr, item.coldkey); cell(tr, TeutonicDashboardV1.verdictLabel(item.verdict), "verdict " + (item.verdict || ""), item.error_message); cell(tr, metric(item.mu_hat)); cell(tr, metric(item.lcb)); cell(tr, metric(item.avg_king_loss, 4)); cell(tr, metric(item.avg_challenger_loss, 4)); var when = age(item.timestamp) + (finite(item.wall_time_s) == null ? "" : " (" + metric(item.wall_time_s, 0) + "S)"); cell(tr, when, "", date(item.timestamp)); makeHistoryRowExpandable(tr, details, item, detailKey); body.appendChild(tr); body.appendChild(details); });
  }
  function renderReigns() {
    var body = el("reigns-body"), rows = [];
    competitionSources().forEach(function (source) {
      var d = source.dashboard;
      if (!d) return;
      if (source.key === "main") {
        var seenHotkeys = new Set(), count = 0;
        (d.king_chain || []).slice().sort(function (a, b) { return (b.reign_number || 0) - (a.reign_number || 0); }).forEach(function (item) {
          var key = item.hotkey || "reign:" + item.reign_number;
          if (count < 2 && !seenHotkeys.has(key)) {
            seenHotkeys.add(key); count++;
            rows.push(Object.assign({}, item, { competition: source.key }));
          }
        });
      } else if (d.king) {
        rows.push(Object.assign({}, d.king, d.king_payout || {}, { competition: source.key }));
      } else {
        rows.push({ competition: source.key, awaitingWinner: true });
      }
    });
    renderCompetitionStatus("reigns-status");
    var kingCount = rows.filter(function (item) { return !item.awaitingWinner; }).length;
    text("reign-count", kingCount + (kingCount === 1 ? " KING" : " KINGS"));
    if (!rows.length) return emptyRow(body, 11, "NO REIGNS YET"); clear(body);
    rows.forEach(function (item, index) {
      var tr = document.createElement("tr"); cell(tr, index + 1); cell(tr, item.competition.toUpperCase());
      if (item.awaitingWinner) {
        tr.className = "muted";
        ["--", "--", "--", "--", "AWAITING FIRST WINNER", "--", "--", "--", "--"].forEach(function (value) { cell(tr, value); });
        body.appendChild(tr); return;
      }
      cell(tr, "#" + number(item.reign_number)); cell(tr, item.uid); hotkeyCell(tr, item.hotkey); coldkeyCell(tr, item.coldkey); var modelCell = cell(tr, identity(item), "", item.model_digest); if (item.model_reference) { var link = document.createElement("a"); link.href = new URL(item.model_reference + "manifest.json", MODEL_STORAGE_BASE).href; link.target = "_blank"; link.rel = "noopener"; link.textContent = identity(item); link.title = "Open model manifest"; modelCell.textContent = ""; modelCell.appendChild(link); } cell(tr, percent(item.weight)); cell(tr, metric(item.alpha_per_hour, 3)); cell(tr, usd(item.usd_per_hour)); cell(tr, date(item.crowned_at)); body.appendChild(tr);
    });
  }
  function renderWeightStatus(d) {
    var weight = d.weight_status || {}; text("weight-state", String(weight.state || weight.latest_attempt_state || "NOT SCHEDULED").toUpperCase()); text("weight-block", number(weight.latest_finalized_block || weight.last_attempted_block)); text("weight-next", number(weight.next_due_block)); text("weight-finalized", date(weight.finalized_at));
  }
  function renderChart(d) {
    var svg = el("loss-chart"), tooltip = el("dataset-change-tooltip");
    tooltip.hidden = true;
    var amount = finite(el("smooth-slider").value) || 0;
    var smoothLabel = amount > 0 ? "SMOOTH " + smoothMode.toUpperCase() + " " + amount.toFixed(2) : "SMOOTH OFF";
    text("smooth-status", smoothLabel);
    var chart = TeutonicDashboardV1.lossChartPresentation(d.history || []), points = chart.points;
    text("chart-count", points.length + (points.length === 1 ? " EVALUATION" : " EVALUATIONS"));
    el("chart-empty").hidden = points.length > 0;
    if (!points.length) { svg.innerHTML = ""; return; }

    var bounds = svg.getBoundingClientRect();
    var W = Math.max(320, Math.round(bounds.width || 1000)), H = Math.max(180, Math.round(bounds.height || 280)), left = 48, right = 8, top = 16, bottom = 42, plotBottom = H - bottom;
    var kingPoints = [];
    var rawChallengers = points.map(function (p) { return finite(p.avg_challenger_loss); });
    points.forEach(function (p, i) { if (p.accepted && p.publication_disposition === "winner" && p.model_identity === "public") kingPoints.push({ index: i, loss: finite(p.avg_challenger_loss) }); });
    var rawKings = kingPoints.map(function (p) { return p.loss; });
    var challengers = smoothSeries(rawChallengers, amount);
    var kings = smoothSeries(rawKings, amount);
    var min = 0, max = chart.maximum;

    function x(i) { return points.length === 1 ? (left + W - right) / 2 : left + i / (points.length - 1) * (W - left - right); }
    function y(v) { var visible = Math.max(min, Math.min(max, v)); return top + (max - visible) / (max - min) * (H - top - bottom); }
    function challengerLine(series) { return series.map(function (value, i) { return x(i).toFixed(1) + "," + y(value).toFixed(1); }).join(" "); }
    function kingLine(series) { return series.map(function (value, i) { return x(kingPoints[i].index).toFixed(1) + "," + y(value).toFixed(1); }).join(" "); }
    var styles = getComputedStyle(document.documentElement), ink = styles.getPropertyValue("--ink").trim(), muted = styles.getPropertyValue("--muted").trim(), paper = styles.getPropertyValue("--paper").trim(), markup = "";
    var yDigits = max - min < .01 ? 4 : max - min < .1 ? 3 : 2;
    for (var tick = 0; tick <= 5; tick++) { var value = min + (max - min) * tick / 5, yy = y(value); markup += '<line x1="' + left + '" y1="' + yy + '" x2="' + (W - right) + '" y2="' + yy + '" stroke="' + muted + '" opacity=".24" stroke-dasharray="3 5"/><line x1="' + (left - 4) + '" y1="' + yy + '" x2="' + left + '" y2="' + yy + '" stroke="' + ink + '"/><text x="' + (left - 9) + '" y="' + (yy + 3) + '" fill="' + muted + '" font-family="Space Mono" font-size="9" text-anchor="end">' + value.toFixed(yDigits) + '</text>'; }
    var xTickCount = Math.min(7, points.length), xTicks = [];
    if (xTickCount === 1) xTicks.push(0); else for (var xTick = 0; xTick < xTickCount; xTick++) { var xIndex = Math.round(xTick * (points.length - 1) / (xTickCount - 1)); if (xTicks.indexOf(xIndex) === -1) xTicks.push(xIndex); }
    xTicks.forEach(function(index) { var xx = x(index); markup += '<line x1="' + xx + '" y1="' + top + '" x2="' + xx + '" y2="' + plotBottom + '" stroke="' + muted + '" opacity=".18" stroke-dasharray="2 6"/><line x1="' + xx + '" y1="' + plotBottom + '" x2="' + xx + '" y2="' + (plotBottom + 4) + '" stroke="' + ink + '"/><text x="' + xx + '" y="' + (plotBottom + 17) + '" fill="' + muted + '" font-family="Space Mono" font-size="9" text-anchor="middle">' + (index + 1) + '</text>'; });
    markup += '<line x1="' + left + '" y1="' + top + '" x2="' + left + '" y2="' + plotBottom + '" stroke="' + ink + '" opacity=".8"/><line x1="' + left + '" y1="' + plotBottom + '" x2="' + (W - right) + '" y2="' + plotBottom + '" stroke="' + ink + '" opacity=".8"/>';
    if (amount > 0 && points.length > 1) markup += '<polyline points="' + challengerLine(rawChallengers) + '" fill="none" stroke="' + muted + '" stroke-width="1" opacity=".24" stroke-dasharray="2 5"/>';
    if (amount > 0 && kingPoints.length > 1) markup += '<polyline points="' + kingLine(rawKings) + '" fill="none" stroke="' + ink + '" stroke-width="1" opacity=".2" stroke-dasharray="2 5"/>';
    if (points.length > 1) markup += '<polyline points="' + challengerLine(challengers) + '" fill="none" stroke="' + muted + '" stroke-width="1.5" stroke-dasharray="6 5"/>';
    if (kingPoints.length > 1) markup += '<polyline points="' + kingLine(kings) + '" fill="none" stroke="' + ink + '" stroke-width="2"/>';
    var pointRadius = TeutonicDashboardV1.graphPointRadius(points.length, W - left - right, 0.75, 2.5), kingPointRadius = Math.min(3, pointRadius + 0.5);
    challengers.forEach(function (value, i) { markup += '<circle cx="' + x(i) + '" cy="' + y(value) + '" r="' + pointRadius.toFixed(2) + '" fill="' + paper + '" stroke="' + muted + '"/>'; });
    kings.forEach(function (value, i) { markup += '<circle cx="' + x(kingPoints[i].index) + '" cy="' + y(value) + '" r="' + kingPointRadius.toFixed(2) + '" fill="' + ink + '"/>'; });
    svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.innerHTML = markup;
    var namespace = "http://www.w3.org/2000/svg", datasetChanges = TeutonicDashboardV1.datasetChangePresentation(points, d.dataset_versions);
    function tooltipContent(change) {
      clear(tooltip);
      var heading = document.createElement("strong"), versions = document.createElement("span"), list = document.createElement("ul");
      heading.textContent = "DATASET CHANGED";
      versions.textContent = change.fromLabel + " [" + change.fromVersion.slice(0, 8) + "] → " + change.toLabel + " [" + change.toVersion.slice(0, 8) + "]";
      change.changes.forEach(function(description) { var item = document.createElement("li"); item.textContent = description; list.appendChild(item); });
      tooltip.appendChild(heading); tooltip.appendChild(versions); tooltip.appendChild(list);
    }
    function positionTooltip(clientX) {
      var shell = svg.parentElement, bounds = shell.getBoundingClientRect(), requested = clientX - bounds.left + 10;
      tooltip.style.top = "8px"; tooltip.style.left = "8px"; tooltip.hidden = false;
      tooltip.style.left = Math.max(8, Math.min(requested, bounds.width - tooltip.offsetWidth - 8)) + "px";
    }
    function showTooltip(change, clientX) { tooltipContent(change); positionTooltip(clientX); }
    function hideTooltip() { tooltip.hidden = true; }
    datasetChanges.forEach(function(change) {
      var marker = document.createElementNS(namespace, "g"), line = document.createElementNS(namespace, "line"), hit = document.createElementNS(namespace, "line"), flag = document.createElementNS(namespace, "rect"), label = document.createElementNS(namespace, "text"), xx = (x(change.index - 1) + x(change.index)) / 2, flagWidth = 68, flagX = Math.max(left, Math.min(xx - flagWidth / 2, W - right - flagWidth));
      marker.setAttribute("class", "dataset-change-marker"); marker.setAttribute("tabindex", "0"); marker.setAttribute("role", "img"); marker.setAttribute("aria-label", "Dataset changed from " + change.fromLabel + " to " + change.toLabel + ". " + change.changes.join(". "));
      [line, hit].forEach(function(node) { node.setAttribute("x1", xx); node.setAttribute("x2", xx); node.setAttribute("y1", top); node.setAttribute("y2", plotBottom); });
      line.setAttribute("class", "dataset-change-line"); line.setAttribute("stroke", muted); line.setAttribute("stroke-width", "1.5"); line.setAttribute("stroke-dasharray", "5 4");
      hit.setAttribute("class", "dataset-change-hit");
      flag.setAttribute("class", "dataset-change-flag"); flag.setAttribute("x", flagX); flag.setAttribute("y", 2); flag.setAttribute("width", flagWidth); flag.setAttribute("height", 14); flag.setAttribute("fill", paper); flag.setAttribute("stroke", muted);
      label.setAttribute("class", "dataset-change-label"); label.setAttribute("x", flagX + flagWidth / 2); label.setAttribute("y", 12); label.setAttribute("fill", muted); label.setAttribute("text-anchor", "middle"); label.textContent = "DATASET Δ";
      marker.appendChild(line); marker.appendChild(hit); marker.appendChild(flag); marker.appendChild(label); svg.appendChild(marker);
      marker.addEventListener("pointerenter", function(event) { showTooltip(change, event.clientX); });
      marker.addEventListener("pointermove", function(event) { positionTooltip(event.clientX); });
      marker.addEventListener("pointerleave", hideTooltip);
      marker.addEventListener("focus", function() { var markerBounds = marker.getBoundingClientRect(); showTooltip(change, markerBounds.left + markerBounds.width / 2); });
      marker.addEventListener("blur", hideTooltip);
    });
  }
  function render(d) { TeutonicDashboardV1.validate(d); lastPayload = d; mainQueueFailed = false; renderHeader(d); renderReigns(); renderChart(d); renderSharedEvaluation(); renderHistory(); renderWeightStatus(d); text("last-refresh", "LAST REFRESH " + new Date().toLocaleTimeString()); el("error-banner").hidden = true; }
  async function poll() { try { var response = await fetch(ENDPOINT + "?t=" + Date.now(), { cache: "no-store" }); if (!response.ok) throw new Error("dashboard request returned HTTP " + response.status); render(await response.json()); } catch (error) { mainQueueFailed = true; renderSharedEvaluation(); renderReigns(); renderHistory(); var banner = el("error-banner"); banner.textContent = "DATA REFRESH FAILED — " + error.message + (lastPayload ? " — SHOWING LAST GOOD PUBLICATION" : ""); banner.hidden = false; } }
  function setTheme(theme) { document.documentElement.dataset.theme = theme; el("theme-toggle").textContent = theme === "dark" ? "LIGHT" : "DARK"; if (lastPayload) renderChart(lastPayload); }
  var savedTheme = localStorage.getItem("dashboard-theme"); setTheme(savedTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")); el("theme-toggle").addEventListener("click", function () { var next = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; localStorage.setItem("dashboard-theme", next); setTheme(next); });
  var smoothSlider = el("smooth-slider"), savedSmoothing = localStorage.getItem("smoothing");
  if (savedSmoothing != null && finite(savedSmoothing) != null) smoothSlider.value = savedSmoothing;
  function updateSmoothControls() { var value = Number(smoothSlider.value); text("smooth-value", value.toFixed(2).replace(/0$/, "").replace(/\.$/, "")); el("smooth-mode-toggle").textContent = smoothMode === "normal" ? "NORMAL" : "LOWESS"; el("smooth-mode-toggle").title = smoothMode === "normal" ? "Normal EMA smoothing; click for LOWESS" : "LOWESS smoothing; click for normal EMA"; }
  updateSmoothControls();
  smoothSlider.addEventListener("input", function () { localStorage.setItem("smoothing", smoothSlider.value); updateSmoothControls(); if (lastPayload) renderChart(lastPayload); });
  el("smooth-mode-toggle").addEventListener("click", function () { smoothMode = smoothMode === "lowess" ? "normal" : "lowess"; localStorage.setItem("smoothMode", smoothMode); updateSmoothControls(); if (lastPayload) renderChart(lastPayload); });
  el("history-errors-toggle").addEventListener("click", function () { historyShowErrors = !historyShowErrors; renderHistory(); });
  var historyFilters = el("history-competition-filters").querySelectorAll("button");
  historyFilters.forEach(function (button) {
    button.addEventListener("click", function () {
      historyCompetition = button.dataset.competition;
      historyFilters.forEach(function (filter) { filter.setAttribute("aria-pressed", String(filter === button)); });
      renderHistory();
      el("history-body").closest(".history-wrap").scrollTop = 0;
    });
  });
  el("benchmark-history-toggle").addEventListener("click", function () { benchmarkHistoryVisible = !benchmarkHistoryVisible; if (benchmarkPayload) renderBenchmarks(benchmarkPayload); });
  var datasetHelp = el("competition-dataset-help"), datasetHelpButton = datasetHelp.querySelector("button"), datasetHelpTooltip = el("competition-dataset-tooltip");
  function showDatasetHelp() { datasetHelpTooltip.hidden = false; }
  datasetHelp.addEventListener("pointerenter", showDatasetHelp);
  datasetHelp.addEventListener("pointerleave", function () { if (document.activeElement !== datasetHelpButton) datasetHelpTooltip.hidden = true; });
  datasetHelpButton.addEventListener("focus", showDatasetHelp);
  datasetHelpButton.addEventListener("click", showDatasetHelp);
  datasetHelpButton.addEventListener("blur", function () { if (!datasetHelp.matches(":hover")) datasetHelpTooltip.hidden = true; });
  document.addEventListener("keydown", function (event) { if (event.key === "Escape") datasetHelpTooltip.hidden = true; });
  var chartResizeFrame = null;
  window.addEventListener("resize", function () { if (chartResizeFrame != null) cancelAnimationFrame(chartResizeFrame); chartResizeFrame = requestAnimationFrame(function () { chartResizeFrame = null; if (lastPayload) renderChart(lastPayload); }); });
  var specialists = ["math", "text", "code"].map(function (key) {
    var state = { key: key, prefix: "/competitions/" + key + "/", dashboard: null, failed: false, polling: false, loadingDataset: false, dataset: null, datasetFailed: false, datasetRow: document.createElement("tr") };
    state.datasetRow.id = "dataset-" + key;
    el("competition-datasets").appendChild(state.datasetRow);
    renderCompetitionDataset(state);
    return state;
  });
  async function pollSpecialist(state) {
    if (state.polling) return;
    state.polling = true;
    try {
      var d = await fetchJson(state.prefix + "dashboard.json");
      TeutonicDashboardV1.validate(d); state.dashboard = d; state.failed = false;
    } catch (error) { state.failed = true; }
    finally { state.polling = false; renderSharedEvaluation(); renderReigns(); renderHistory(); }
  }
  function renderCompetitionDataset(state) {
    var row = state.datasetRow, manifest = state.dataset, view = manifest ? TeutonicDashboardV1.datasetPresentation(manifest) : null;
    clear(row);
    var heading = document.createElement("th"); heading.scope = "row"; heading.textContent = state.key.toUpperCase(); row.appendChild(heading);
    if (!view || state.datasetFailed) {
      var status = document.createElement("span"); status.className = "dataset-sub";
      status.textContent = state.datasetFailed ? (view ? "REFRESH FAILED · SHOWING LAST MIX" : "CONFIGURATION UNAVAILABLE · USE THE MANIFEST LINK") : "LOADING DATASET MIX…";
      heading.appendChild(status);
    }
    ["math", "text", "code"].forEach(function (key) {
      var source = view && view.rows.find(function (item) { return item.name.toLowerCase() === key; });
      datasetCell(row, source ? datasetWeight(source.normalizedWeight) : "--", source ? number(source.evalSequences) + " SEQ" : "");
    });
    cell(row, view ? number(view.evalN) : "--");
    cell(row, manifest ? "> " + metric(manifest.delta_threshold, 3) : "--");
    datasetCell(row, "MANIFEST", "", state.prefix + "datasets/manifest.json");
  }
  async function loadSpecialistDataset(state) {
    if (state.loadingDataset) return;
    state.loadingDataset = true;
    try {
      var manifest = await fetchJson(state.prefix + "datasets/manifest.json");
      TeutonicDashboardV1.datasetPresentation(manifest);
      state.dataset = manifest; state.datasetFailed = false;
    } catch (error) { state.datasetFailed = true; }
    finally { state.loadingDataset = false; renderCompetitionDataset(state); }
  }
  function pollSpecialists() { specialists.forEach(pollSpecialist); }
  function loadSpecialistDatasets() { specialists.forEach(loadSpecialistDataset); }
  pollSpecialists(); setInterval(pollSpecialists, POLL_MS);
  loadSpecialistDatasets(); setInterval(loadSpecialistDatasets, DATASET_POLL_MS);
  poll(); setInterval(poll, POLL_MS);
  loadDatasetManifest(); setInterval(loadDatasetManifest, DATASET_POLL_MS);
  loadBenchmarks(); setInterval(loadBenchmarks, BENCHMARK_POLL_MS);
})();
