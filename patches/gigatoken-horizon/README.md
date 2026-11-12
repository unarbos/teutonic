# Native Horizon support for Gigatoken 0.11.0

This local patch adds IFM K2-Horizon's exact pre-tokenization regex to Gigatoken's
Rust backend. It keeps the original tokenizer JSON, vocabulary, BPE merges,
normalization, and special tokens. Unlike the temporary subset adapter, it supports
combining marks and U+200C/U+200D directly, without routing documents to Hugging Face.

Horizon extends Olmo3's letter run with marks and joiners. Those characters also
remain legal inside its punctuation runs, so simply treating them as letters in
all branches would change the tokenizer. The patch extends only the letter-run
predicate and prefix lookup, preserving the original punctuation and whitespace
rules. Pure ASCII segments reuse upstream Olmo3's optimized scanner. Unicode
segments use a native scalar scanner with the existing word-at-a-time letter scan.

The source archive URL and SHA-256 are in `source-pin.json`. `horizon.patch`
produces version `0.11.0+teutonic.horizon.1`. The upstream MIT license is retained.
Nothing is published upstream or to R2.

To build with Rust installed (nightly-2026-11-10) and Maturin available:

```bash
uv pip install --python .venv/bin/python 'maturin>=1.14.1,<2'
.venv/bin/python scripts/build_horizon_gigatoken.py \
  --build-dir /tmp/gigatoken-horizon-build \
  --output-dir artifacts/gigatoken-horizon/wheels
uv pip install --python .venv/bin/python \
  artifacts/gigatoken-horizon/wheels/gigatoken-0.11.0+teutonic.horizon.1-*.whl
```

Build on the target host or use a wheel compatible with its platform. The evaluator
extra's `gigatoken==0.11.0` requirement accepts this PEP 440 local version; install the
patched wheel after the evaluator dependencies. Unpatched Gigatoken rejects Horizon's
regex rather than changing its tokenization silently.

Verify both **local, pinned** tokenizers before GPU testing:

```bash
.venv/bin/python scripts/verify_packed_tokenizers.py \
  --mimo-snapshot /path/to/king \
  --horizon-snapshot /path/to/pinned/horizon \
  --report artifacts/packed-text-tokenizer-verification.json
```

Comparisons happen in this verification command and the tests. Production encodes
each document once. `TEUTONIC_TOKENIZER_BACKEND=reference` and
`TEUTONIC_REFERENCE_TOKENIZER_BACKEND=reference` are explicit diagnostic alternatives;
there is no automatic per-document fallback or every-batch correctness comparison.

Local verification matched every token ID on 2,047 MiMo stress documents and
2,647 Horizon stress documents, plus the 25 recovered source documents.
For those 25 documents (422,701 UTF-8 bytes), median warm encoding over five
repeats measured MiMo at 49.0 ms reference / 4.6 ms Gigatoken (10.6×), and
Horizon at 49.3 ms / 5.8 ms (8.6×). Horizon's first batch was slower with Gigatoken
(104.7 ms versus 71.5 ms); these figures exclude tokenizer construction and GPU
inference. They are a small repeated-corpus benchmark, not full-evaluator throughput.

Reproduce with `scripts/benchmark_packed_tokenizers.py --documents sample.jsonl
--mimo-snapshot /path/to/king --horizon-snapshot /path/to/horizon --report result.json`.
Each local JSONL row must contain a `text` string. This is a benchmark input;
the production dataset remains packed UTF-8 with byte-offset indexes.
