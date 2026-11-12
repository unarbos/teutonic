#!/usr/bin/env python3
"""Compare native Gigatoken to local pinned tokenizers. No network operations."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluator.text_pipeline import TextTokenizer


def corpus():
    examples = [
        '', 'x', 'Hello world!', "can't I'LL we're", '\t  \n trailing  ', '\r\n\r\n',
        'Zażółć gęślą jaźń. 你好 🌍', 'é cafe\u0301', 'İ ᾀ', 'क्\u200dष',
        '👨\u200d👩\u200d👦', '中文 中文。１２３４５６', ' ١٢٣٤٥٦٧\n\r\n',
        '\u0301\u0308abc', 'abc\u200cdef', 'abc\u200ddef', 'नमस्ते',
        '\x00\x01\x7f', '<|ifm|endoftext|>', '<|im_end|>', 'abc'*16000,
    ]
    rng = random.Random(431)
    pieces = ["hello", "don't", '\r\n', ' ', '\t', 'é', 'cafe\u0301', '你好', '🌍',
              '১২৩৪', 'a\u0301', '\u200d', 'क', '्', '<|ifm|endoftext|>', '12345', ':-)']
    examples += [''.join(rng.choices(pieces, k=rng.randint(1,100))) for _ in range(1000)]
    # Exercise ordinary text as well as difficult Unicode paths.
    examples += [''.join(rng.choices(pieces[:9]+pieces[-2:], k=50)) for _ in range(1000)]
    return examples


def verify(snapshot, eos):
    raw = (snapshot/'tokenizer.json').read_bytes()
    texts = corpus()
    texts += [x['content'] for x in json.loads(raw)['added_tokens']]
    tokenizer = TextTokenizer(snapshot, eos_token_id=eos)
    token_count = 0
    for offset in range(0, len(texts), 64):
        batch = texts[offset:offset+64]
        actual = tokenizer.encode(batch)
        expected = [row.ids+[eos] for row in tokenizer.reference.encode_batch(batch, add_special_tokens=False)]
        if actual != expected:
            raise ValueError('adapter differs from pinned tokenizer')
        token_count += sum(map(len, actual))
    return {'tokenizer_sha256': hashlib.sha256(raw).hexdigest(), 'documents': len(texts),
            'input_tokens': token_count, 'exact_match': True, **tokenizer.status()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mimo-snapshot', type=Path, required=True)
    p.add_argument('--horizon-snapshot', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    result = {'mimo': verify(args.mimo_snapshot, 151645), 'horizon': verify(args.horizon_snapshot, 1)}
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
