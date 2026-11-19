(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.HorizonHistoryData = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';
  function evaluated(row) {
    // Packed-document verdicts store their completed population in n_sequences.
    return row.n_sequences_evaluated == null ? row.n_sequences : row.n_sequences_evaluated;
  }
  function main2Kings(payload) {
    var seen = new Set(), reigns = payload.king_chain || [];
    return (payload.history || []).filter(function (row) {
      // Accepted uploads awaiting promotion are not kings. Never substitute the
      // old king's loss, filtered deficit score, or native-token loss here.
      return row.accepted === true && row.publication_disposition === 'winner' &&
        row.model_identity === 'public' && row.loss_unit === 'bits/byte' &&
        row.challenge_id && row.challenger_digest && row.early_stopped !== true &&
        Number.isFinite(row.avg_challenger_loss) && Number.isFinite(row.avg_reference_loss) &&
        row.avg_challenger_loss >= 0 && row.avg_reference_loss >= 0 &&
        Number.isFinite(Date.parse(row.timestamp)) && row.n_sequences > 0 &&
        evaluated(row) === row.n_sequences;
    }).map(function (row) {
      var reign = reigns.find(function (r) { return r.model_digest === row.challenger_digest; });
      return {
        competition: 'main2', challenge_id: row.challenge_id,
        model: row.challenger_repo || row.challenger_digest, model_digest: row.challenger_digest,
        timestamp: reign && reign.crowned_at || row.timestamp,
        documents: evaluated(row), expected_documents: row.n_sequences,
        coverage: 1, complete: true, king: row.avg_challenger_loss,
        horizon: row.avg_reference_loss, gap: row.avg_challenger_loss - row.avg_reference_loss,
        dataset_version: row.dataset_version,
        source: 'Published MAIN2 coronation evaluation; all selected documents'
      };
    }).sort(function (a, b) {
      return Date.parse(a.timestamp) - Date.parse(b.timestamp) || a.challenge_id.localeCompare(b.challenge_id);
    }).filter(function (row) {
      if (seen.has(row.challenge_id)) return false;
      seen.add(row.challenge_id); return true;
    });
  }
  function merge(historical, live) {
    // Competition is part of identity: the same model can hold MAIN and MAIN2.
    var seen = new Set();
    return historical.concat(live).filter(function (row) {
      var key = (row.competition || 'main') + ':' + (row.challenge_id || row.model_digest);
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
  }
  return {main2Kings: main2Kings, merge: merge};
});
