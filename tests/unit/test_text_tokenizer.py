import json
from types import SimpleNamespace

import pytest

from teutonic.evaluator.text_pipeline import TextTokenizer

HORIZON_SPLIT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def test_native_horizon_matches_original_bpe_without_runtime_reference(tmp_path):
    pytest.importorskip('tokenizers')
    pytest.importorskip('gigatoken')
    from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(HORIZON_SPLIT), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    texts = ['hello world', 'a\u0308', 'ka\u0301', 'a\u200cb', 'a\u200db', 'क्\u200dष',
             '<eos> café 12345\n', 'hello hello', '🧑\u200d💻', '\u0301\u0308abc', '',
             '!!\u0301abc', ' \u200dfoo', '\u200c\u200d\u0301', '\t\u0301a', '\n\u0301a']
    tok.train_from_iterator(texts, trainer=trainers.BpeTrainer(vocab_size=350,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<eos>']))
    tok.save(str(tmp_path/'tokenizer.json'))
    (tmp_path/'config.json').write_text(json.dumps({'eos_token_id': 0}))
    wrapped = TextTokenizer(tmp_path)
    expected = [t.ids+[0] for t in tok.encode_batch(texts, add_special_tokens=False)]
    def forbidden(*args, **kwargs):
        raise AssertionError('production must not run the reference tokenizer')
    wrapped.reference = SimpleNamespace(encode_batch=forbidden)
    assert wrapped.encode(texts) == expected
    assert wrapped.status()['gigatoken_documents'] == len(texts)
    assert wrapped.status()['reference_documents'] == 0
    # Production MiMo configs may omit EOS; the evaluator supplies its pinned ID.
    (tmp_path/'config.json').unlink()
    assert TextTokenizer(tmp_path, eos_token_id=0).encode(texts) == expected
