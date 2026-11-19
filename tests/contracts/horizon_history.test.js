'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '../../website');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const chart = fs.readFileSync(path.join(root, 'horizon-history.js'), 'utf8');
const data = JSON.parse(fs.readFileSync(path.join(root, 'horizon-history.json'), 'utf8'));
const panels = {'loss-history-panel': {hidden: false}, 'horizon-history-panel': {hidden: true}};
const tabs = ['performance-heading', 'horizon-history-tab'].map((id, i) => ({
  id, attrs: {'aria-controls': Object.keys(panels)[i]}, events: {},
  setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
  addEventListener(k, fn) { this.events[k] = fn; }, focus() { this.focused = true; }
}));
let originalDraws = 0, horizonDraws = 0;
vm.runInNewContext(app.slice(app.indexOf('  var chartTabs ='), app.indexOf('  el("history-errors-toggle")')), {
  document: {querySelectorAll: () => tabs}, el: id => panels[id], lastPayload: {},
  renderChart: () => originalDraws++, window: {HorizonComparison: {render: () => horizonDraws++}}
});
tabs[1].events.click();
assert.equal(panels['loss-history-panel'].hidden, true);
assert.equal(panels['horizon-history-panel'].hidden, false);
assert.equal(tabs[1].attrs['aria-selected'], 'true');
assert.equal(horizonDraws, 1);
tabs[1].events.keydown({key: 'ArrowLeft', preventDefault() {}});
assert.equal(panels['loss-history-panel'].hidden, false);
assert.equal(panels['horizon-history-panel'].hidden, true);
assert.equal(tabs[0].tabIndex, 0);
assert.equal(tabs[1].tabIndex, -1);
assert.equal(tabs[0].focused, true);
assert.equal(originalDraws, 1);
for (const id of [...Object.keys(panels), ...tabs.map(t => t.id)]) assert.ok(html.includes('id="' + id + '"'));
assert.ok(!html.includes('Average loss · bits/byte · promotion order · lower is better'));
assert.ok(!chart.includes('trend-readout'));
assert.ok(!chart.includes('trend-coverage'));
assert.equal(data.unit, 'bits/byte');
assert.equal(data.kings.length, 14);
for (const king of data.kings) {
  assert.equal(king.complete, true);
  assert.equal(king.documents, king.expected_documents);
  assert.ok(Number.isFinite(king.king) && Number.isFinite(king.horizon));
  assert.ok(Math.abs(king.king - king.horizon - king.gap) < 1e-12);
}
console.log('History tabs switch with mouse and keyboard; the approved complete, unfiltered 14-king snapshot is bundled.');

const historyData = require('../../website/horizon-history-data.js');
function winner(overrides) {
  return Object.assign({accepted: true, publication_disposition: 'winner', model_identity: 'public',
    loss_unit: 'bits/byte', challenge_id: 'new-main2', challenger_digest: 'model-a', challenger_repo: 'MAIN2 king',
    avg_challenger_loss: .25, avg_king_loss: .8, avg_reference_loss: .22,
    avg_challenger_token_loss: .9, n_sequences: 200, n_sequences_evaluated: null,
    early_stopped: false, timestamp: '2026-11-20T10:00:00Z'}, overrides);
}
const first = winner();
const second = winner({challenge_id: 'newer-main2', challenger_digest: 'model-b', timestamp: '2026-11-21T10:00:00Z'});
const payload = {history: [second, first, first,
  winner({accepted: false}), winner({publication_disposition: 'pending'}),
  winner({avg_reference_loss: null}), winner({loss_unit: 'nats/token'}),
  winner({early_stopped: true}), winner({n_sequences_evaluated: 100})],
  king_chain: [{model_digest: 'model-a', crowned_at: '2026-11-20T10:05:00Z'}]};
const liveKings = historyData.main2Kings(payload);
assert.equal(liveKings.length, 2);
assert.equal(liveKings[0].challenge_id, first.challenge_id);
assert.equal(liveKings[0].timestamp, '2026-11-20T10:05:00Z');
assert.equal(liveKings[0].king, .25); // Winning challenger, not baseline king or token loss.
assert.equal(liveKings[0].horizon, .22);
assert.equal(liveKings[0].documents, 200);
const merged = historyData.merge(data.kings, liveKings);
assert.equal(merged.length, 16);
assert.deepEqual(historyData.merge(merged, liveKings), merged); // Refresh cannot duplicate kings.
assert.equal(historyData.merge([{...liveKings[0], competition: 'main'}], liveKings).length, 3);
assert.ok(app.includes('if (state.key === "main2") window.HorizonComparison.updateMain2(d);'));
assert.ok(!chart.includes("'>#'+(i+1)"));
console.log('New MAIN2 crowned winners append automatically; rejected/incomplete/missing-reference rows are excluded; no numbered axis labels.');
