from __future__ import annotations
import copy
import hashlib
import re
import unittest
from collections import Counter

from teutonic.evaluation.categories import DEFAULT_RULES_PATH, category_of, load_rules
from teutonic.evaluation.configuration import DatasetManifestSnapshot, EvaluationSettings, canonical_manifest_bytes, pretokenized_dataset_request
from teutonic.evaluation.protocol_v2 import EvaluationRequestV2, ProtocolValidationError
from tests.unit.test_evaluator_protocol_v2 import request_payload


class CategoryCoverageTests(unittest.TestCase):
    def settings(self, n=200):
        shards = []
        for category in range(7):
            for index in range(category + 1):
                shards.append(dict(key=f'code-reasoning__cat{category}--{index}.npy',
                                   sha256=hashlib.sha256(f'{category}:{index}'.encode()).hexdigest(),
                                   n_tokens=2048*1000, size_bytes=8192000))
        manifest = {'shards':shards}
        snapshot = DatasetManifestSnapshot('code-reasoning', 'https://example.com/manifest.json',
                   hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(), 1.0, manifest)
        return EvaluationSettings('a'*64, 'test', n, 0.0125, (snapshot,), 4)

    def request(self, seed, n=200):
        return pretokenized_dataset_request(self.settings(n), block_hash='0x'+f'{seed:064x}',
                                           hotkey='hotkey', seq_len=2048)

    def test_default_rules_cover_every_category_across_seeds(self):
        rule=load_rules(DEFAULT_RULES_PATH)['code-reasoning']
        signatures=set(); allocations=[]
        for seed in range(10):
            request=self.request(seed)
            shards=request['sources'][0]['shards']
            counts=Counter()
            for shard in shards:
                counts[category_of(rule,shard['url'])]+=shard['target_sequences']
            self.assertEqual(set(counts),{f'cat{i}' for i in range(7)})
            self.assertTrue(all(n>0 for n in counts.values()))
            self.assertEqual(sum(counts.values()),200)
            self.assertGreaterEqual(len(shards),7)
            allocations.append(counts)
            signatures.add(tuple(s['url'] for s in shards))
            self.assertEqual(request,self.request(seed))
        self.assertTrue(all(c==allocations[0] for c in allocations))
        self.assertGreater(len(signatures),1)

    def test_insufficient_sample_count_fails_instead_of_omitting_categories(self):
        with self.assertRaisesRegex(ValueError,'cover all 7 categories'):
            self.request(1,n=6)

    def test_source_file_is_only_used_for_category_not_download_url(self):
        settings=self.settings()
        manifest=copy.deepcopy(settings.manifests[0].manifest)
        for shard in manifest['shards']:
            shard['source_file']=shard['key'].replace('.npy','.parquet')
        snapshot=DatasetManifestSnapshot('code-reasoning','https://example.com/manifest.json',
                  hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),1.0,manifest)
        settings=EvaluationSettings('a'*64,'test',200,0.0125,(snapshot,),4)
        result=pretokenized_dataset_request(settings,block_hash='0x'+'a'*64,hotkey='hotkey',seq_len=2048)
        self.assertTrue(all(s['url'].endswith('.npy') for s in result['sources'][0]['shards']))

    def test_protocol_rejects_partial_or_inconsistent_shard_quotas(self):
        payload=request_payload()
        source=payload['dataset']['sources'][0]
        source['shards'].append(copy.deepcopy(source['shards'][0]))
        source['shards'][0]['target_sequences']=100
        with self.assertRaisesRegex(ProtocolValidationError,'all present or all absent'):
            EvaluationRequestV2.from_mapping(payload)
        source['shards'][1]['target_sequences']=100
        with self.assertRaisesRegex(ProtocolValidationError,'sum to source target'):
            EvaluationRequestV2.from_mapping(payload)
        source['shards'][0]['target_sequences']=12500
        source['shards'][1]['target_sequences']=12500
        EvaluationRequestV2.from_mapping(payload)


if __name__=='__main__':unittest.main()
