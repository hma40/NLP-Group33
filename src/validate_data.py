"""Independently reload CSVs and verify schema, hashes and cross-split separation."""
import json

import pandas as pd

try:
    from .data_common import PROCESSED, STATS, key, validate
    from .dataset_catalog import DATASETS, TASKS
except ImportError:
    from data_common import PROCESSED, STATS, key, validate
    from dataset_catalog import DATASETS, TASKS


def main():
    for name in DATASETS:
        parts = {s: pd.read_csv(PROCESSED / name / f'{s}.csv', keep_default_na=False) for s in ['train', 'dev', 'test']}
        validate(parts)
        audit = json.loads((STATS / (name.replace('/', '_') + '.json')).read_text(encoding='utf-8'))
        seen_links = {}
        for split, data in parts.items():
            assert {'id', 'text', 'label', 'irony_type', 'context', 'source', 'platform'}.issubset(data.columns)
            assert len(data) == audit['splits'][split]['count']
            assert data.label.sum() == audit['splits'][split]['positive']
            assert data.text.ne('').all(), f'{name}/{split}: empty model text'
            if name in ['hahackathon', 'semeval2022_idioms']:
                assert data.task.eq(TASKS[name][0]).all()
                assert data.positive_label.eq(TASKS[name][1]).all()
            if name == 'semeval2022_idioms':
                assert data.label.eq(1 - data.source_label).all(), 'Reversed idiom labels'
            for context in data.context:
                assert isinstance(json.loads(context), list)
            # Recheck observable links independently of the saved component IDs.
            for row in data.to_dict('records'):
                texts = [row['text'], row.get('rephrasing', '')]
                texts += [t for t in json.loads(row['context']) if len(key(t)) >= 20]
                if len(key(row.get('following_context', ''))) >= 20:
                    texts.append(row['following_context'])
                links = ['text:' + key(t) for t in texts if key(t)]
                for col in ['conversation_id', 'author', 'template_id']:
                    value = str(row.get(col, '')).strip()
                    if value and value not in {'nan', '[deleted]', '[removed]'}:
                        links.append(col + ':' + value)
                for link in links:
                    assert link not in seen_links or seen_links[link] == split, f'{name}: shared input/metadata link'
                    seen_links[link] = split
            import hashlib
            assert hashlib.sha256((PROCESSED / name / f'{split}.csv').read_bytes()).hexdigest() == audit['splits'][split]['csv_sha256']
        print(f'{name}: schema, labels, IDs, groups, normalized text, counts and hashes OK')
    try:
        from .preprocess import semeval2018, semeval2022, figlang, sample_sequences
        from .data_common import RAW
    except ImportError:
        from preprocess import semeval2018, semeval2022, figlang, sample_sequences
        from data_common import RAW
    sources = [('semeval2018', semeval2018()), ('semeval2022', semeval2022()),
               ('figlang2020/twitter', figlang('twitter')), ('figlang2020/reddit', figlang('reddit'))]
    for name, (source, _) in sources:
        expected = source.loc[source.partition.eq('test')].sort_values('id')
        actual = pd.read_csv(PROCESSED / name / 'test.csv', keep_default_na=False).sort_values('id')
        for col in ['id', 'original_text', 'label']:
            assert expected[col].tolist() == actual[col].tolist(), f'{name}: official test changed'
        print(f'{name}: every official test ID, original target and label retained')
    selected, _ = sample_sequences(RAW / 'sarc/main/test-balanced.csv.bz2', 2000)
    # Use configured sample size, so validation also supports a user's larger run.
    audit = json.loads((STATS / 'sarc.json').read_text(encoding='utf-8'))
    configured = audit['metadata']['requested_test_sample']
    if configured != 2000:
        selected, _ = sample_sequences(RAW / 'sarc/main/test-balanced.csv.bz2', configured)
    expected = {r['comment_id']: r['label'] for r in selected}
    actual = pd.read_csv(PROCESSED / 'sarc/test.csv', keep_default_na=False)
    assert dict(zip(actual.comment_id, actual.label)) == expected, 'SARC official test sample changed'
    print('sarc: official test sample identifiers and labels verified against source sequences')
    try:
        from .figurative_datasets import hahackathon
    except ImportError:
        from figurative_datasets import hahackathon
    source, _ = hahackathon()
    expected = source.loc[source.partition.eq('test')].sort_values('id')
    actual = pd.read_csv(PROCESSED / 'hahackathon/test.csv', keep_default_na=False).sort_values('id')
    for col in ['id', 'original_text', 'label']:
        assert expected[col].tolist() == actual[col].tolist(), 'HaHackathon official test changed'
    print('hahackathon: every official test ID, original target and label retained')


if __name__ == '__main__':
    main()
