"""Shared minimal cleaning, leakage groups, splitting and aggregate reporting."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'Data' / 'Raw'
PROCESSED = ROOT / 'Data' / 'Processed'
STATS = ROOT / 'results' / 'dataset_statistics'
SEED = 42
MARKERS = re.compile(r'(?i)(?<!\w)#(?:sarcasm|sarcastic|irony|not)\b|(?<!\w)/s\b')


def clean(text):
    return MARKERS.sub('', str(text)).strip()


def key(text):
    """Comparison only: model text retains its case, emoji and punctuation."""
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', str(text).casefold())
    text = re.sub(r'@\w+', ' USER ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    return ' '.join(re.findall(r'\w+', text)) or text.strip()


def digest(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def standardize(frame, source, platform, strip_label_markers=True):
    frame = frame.copy()
    raw_missing = {c: int(v) for c, v in frame.isna().sum().items()}
    frame['source'] = source
    frame['platform'] = platform
    frame['original_text'] = frame['text'].fillna('')
    cleaner = clean if strip_label_markers else lambda value: str(value).strip()
    frame['text'] = frame['original_text'].map(cleaner)
    for col, default in [('context', '[]'), ('irony_type', ''), ('rephrasing', ''),
                         ('conversation_id', ''), ('author', ''), ('parent_comment', '')]:
        if col not in frame:
            frame[col] = default
        frame[col] = frame[col].fillna(default)
    # Context and rephrasing can become model inputs later: strip label markers there too.
    frame['context'] = frame['context'].map(
        lambda s: json.dumps([cleaner(t) for t in json.loads(s)], ensure_ascii=False))
    frame['rephrasing'] = frame['rephrasing'].map(cleaner)
    frame['parent_comment'] = frame['parent_comment'].map(cleaner)
    if not set(frame['label'].dropna().unique()).issubset({0, 1}):
        raise ValueError(f'{source}: labels must be 0/1')
    if frame['label'].isna().any():
        raise ValueError(f'{source}: missing labels')
    frame['label'] = frame['label'].astype(int)
    if frame['id'].duplicated().any():
        raise ValueError(f'{source}: IDs are not unique')
    frame.attrs['raw_missing_values'] = raw_missing
    frame.attrs['strip_label_markers'] = strip_label_markers
    return frame


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def leakage_groups(frame):
    """Connected components keep transitive text/context/author/template links together."""
    uf = UnionFind(len(frame))
    seen = {}
    for i, row in enumerate(frame.to_dict('records')):
        tokens = []
        texts = [row['text'], row['rephrasing']]
        texts += [t for t in json.loads(row['context']) if len(key(t)) >= 20]
        if len(key(row.get('following_context', ''))) >= 20:
            texts.append(row['following_context'])
        for text in texts:
            normalized = key(text)
            if normalized:
                tokens.append('text:' + normalized)
        for col in ['conversation_id', 'author', 'template_id']:
            value = str(row.get(col, '')).strip()
            if value and value not in {'nan', '[deleted]', '[removed]'}:
                tokens.append(col + ':' + value)
        for token in tokens:
            if token in seen:
                uf.join(i, seen[token])
            else:
                seen[token] = i
    # High-similarity character n-grams catch punctuation/spelling variations.
    # Batched sparse products avoid an N-by-N dense matrix.
    unique = frame.drop_duplicates('text')
    indices = unique.index.to_numpy()
    near_pairs = 0
    if len(unique) > 1:
        matrix = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=True).fit_transform(unique.text)
        for start in range(0, len(unique), 128):
            sim = (matrix[start:start + 128] @ matrix.T).tocoo()
            keep = (sim.data >= 0.95) & ((sim.row + start) < sim.col)
            for a, b in zip(sim.row[keep] + start, sim.col[keep]):
                uf.join(int(indices[a]), int(indices[b]))
                near_pairs += 1
    groups = [uf.find(i) for i in range(len(frame))]
    return groups, near_pairs


def split_pool(pool, fraction):
    """Exact stratification for singleton groups; otherwise choose a balanced group split."""
    if pool.group_id.nunique() == len(pool):
        counts = pool.label.value_counts()
        stratify = pool.label if len(counts) == 2 and counts.min() >= 2 else None
        a, b = train_test_split(pool, test_size=fraction, random_state=SEED, stratify=stratify)
        return a.copy(), b.copy()
    if pool.group_id.nunique() < 2:
        raise ValueError('Fewer than two independent groups remain; cannot split without leakage.')
    best = None
    for a, b in GroupShuffleSplit(n_splits=256, test_size=fraction, random_state=SEED).split(pool, groups=pool.group_id):
        left, right = pool.iloc[a], pool.iloc[b]
        if left.label.nunique() < 2 or right.label.nunique() < 2:
            continue
        loss = abs(len(right) / len(pool) - fraction) + abs(right.label.mean() - pool.label.mean())
        if best is None or loss < best[0]:
            best = (loss, left.copy(), right.copy())
    if best is None:
        raise ValueError('Cannot retain both classes in independent groups; inspect template/conversation structure.')
    return best[1:]


def summarize(frame):
    lengths = frame.text.str.len()
    counts = frame.label.value_counts()
    return {'count': len(frame), 'negative': int(counts.get(0, 0)), 'positive': int(counts.get(1, 0)),
            'positive_percent': float(frame.label.mean() * 100),
            'mean_characters': float(lengths.mean()), 'median_characters': float(lengths.median()),
            'mean_words': float(frame.text.str.split().str.len().mean()),
            'context_count': int(frame.context.ne('[]').sum()),
            'irony_categories': sorted(set(frame.irony_type.dropna()) - {''})}


def prepare(frame, name, metadata=None):
    """Keep official test rows; remove conflicting/overlapping material on training side."""
    STATS.mkdir(parents=True, exist_ok=True)
    audit = {'dataset': name, 'seed': SEED, 'metadata': metadata or {}, 'input': summarize(frame),
             'missing_original_text': int(frame.original_text.eq('').sum()),
             'raw_missing_values': frame.attrs.get('raw_missing_values', {}),
             'missing_values': {c: int(v) for c, v in frame.isna().sum().items()},
             'markers_removed_rows': int(frame.original_text.str.contains(MARKERS).sum())
                 if frame.attrs.get('strip_label_markers', True) else 0,
             'trimmed_or_marker_cleaned_rows': int(frame.text.ne(frame.original_text).sum())}
    frame = frame.reset_index(drop=True)
    preserve_dev = (metadata or {}).get('preserve_official_dev', False)
    official = frame.partition.eq('test') | (preserve_dev & frame.partition.eq('dev'))
    frame['_key'] = frame.text.map(key)
    audit['empty_text_rows'] = int(frame.text.str.strip().eq('').sum())
    audit['exact_duplicate_rows'] = int(frame.duplicated('text').sum())
    conflicts = frame.groupby('_key').label.nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    audit['conflicting_text_groups'] = len(conflict_keys)
    audit['conflicting_rows'] = int(frame._key.isin(conflict_keys).sum())
    # Build before exclusions: discarded duplicates can carry additional conversation links.
    groups, pairs = leakage_groups(frame)
    frame['group_id'] = [digest(f'{name}:{g}')[:20] for g in groups]
    audit['near_duplicate_pairs_cosine_095'] = pairs
    remove = ~official & (frame.text.eq('') | frame._key.isin(conflict_keys))
    audit['removed_empty_or_conflicting_train_rows'] = int(remove.sum())
    frame = frame.loc[~remove].reset_index(drop=True)
    # Remove repeated training inputs, retaining all official test rows and metadata in raw files.
    removable = frame.partition.eq('train') if preserve_dev else frame.partition.ne('test')
    remove = removable & frame.duplicated('text')
    audit['removed_duplicate_train_rows'] = int(remove.sum())
    frame = frame.loc[~remove].reset_index(drop=True)
    test = frame.loc[frame.partition.eq('test')].copy()
    pool = frame.loc[frame.partition.ne('test')].copy()
    if len(test):
        if preserve_dev:
            dev = frame.loc[frame.partition.eq('dev')].copy()
            overlap = dev.group_id.isin(test.group_id)
            audit['removed_dev_rows_linked_to_official_test'] = int(overlap.sum())
            dev = dev.loc[~overlap].copy()
            pool = frame.loc[frame.partition.eq('train')].copy()
            held_groups = set(test.group_id) | set(dev.group_id)
        else:
            held_groups = set(test.group_id)
        overlaps = pool.group_id.isin(held_groups)
        audit['removed_train_rows_linked_to_official_test'] = int(overlaps.sum())
        pool = pool.loc[~overlaps]
        if preserve_dev:
            train = pool.copy()
        else:
            train, dev = split_pool(pool, 0.2)
        audit['test_policy'] = 'All loaded official test rows retained; linked training components excluded.'
    else:
        pool, test = split_pool(pool, 0.1)
        train, dev = split_pool(pool, 1 / 9)
        audit['test_policy'] = 'New group-aware 80/10/10 split.'
    parts = {'train': train, 'dev': dev, 'test': test}
    validate(parts)
    out = PROCESSED / name
    out.mkdir(parents=True, exist_ok=True)
    audit['splits'] = {}
    common = ['id', 'text', 'label', 'irony_type', 'context', 'source', 'platform']
    for split, data in parts.items():
        data = data.drop(columns=['_key', 'partition'], errors='ignore').sort_values('id')
        columns = common + [c for c in data if c not in common]
        data[columns].to_csv(out / f'{split}.csv', index=False, encoding='utf-8')
        audit['splits'][split] = summarize(data)
        audit['splits'][split]['csv_sha256'] = hashlib.sha256((out / f'{split}.csv').read_bytes()).hexdigest()
    combined = pd.concat(list(parts.values()))
    audit['processed'] = summarize(combined)
    audit['removed_total'] = audit['input']['count'] - len(combined)
    audit['split_percentages'] = {s: len(d) * 100 / len(combined) for s, d in parts.items()}
    audit['largest_group'] = int(combined.groupby('group_id').size().max())
    audit['official_test_empty_text_rows'] = int(test.text.eq('').sum())
    (STATS / (name.replace('/', '_') + '.json')).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{name}: ' + ', '.join(f'{s}={len(d)}' for s, d in parts.items()), flush=True)
    return audit


def validate(parts):
    for split, data in parts.items():
        if data.id.duplicated().any() or not set(data.label.unique()).issubset({0, 1}):
            raise ValueError(f'{split}: duplicate IDs or invalid labels')
        for other, other_data in parts.items():
            if split >= other:
                continue
            for col in ['id', 'group_id']:
                if set(data[col]) & set(other_data[col]):
                    raise ValueError(f'{split}/{other}: overlapping {col}')
            if set(data.text.map(key)) & set(other_data.text.map(key)):
                raise ValueError(f'{split}/{other}: overlapping normalized target text')
