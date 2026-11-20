"""Prepare each dataset independently. Run from any directory; paths are repository-relative."""
from __future__ import annotations

import argparse
import bz2
import csv
import json
import random

import pandas as pd

try:
    from .data_common import RAW, SEED, digest, prepare, standardize
except ImportError:
    from data_common import RAW, SEED, digest, prepare, standardize


def semeval2018():
    root = RAW / 'semeval2018' / 'datasets'
    frames = []
    for split in ['train', 'test']:
        paths = [root / ('train' if split == 'train' else f'goldtest_Task{task}') /
                 (f'SemEval2018-T3-train-task{task}_emoji.txt' if split == 'train'
                  else f'SemEval2018-T3_gold_test_task{task}_emoji.txt') for task in ['A', 'B']]
        a, b = [pd.read_csv(p, sep='\t', quoting=csv.QUOTE_NONE, dtype=str) for p in paths]
        a.columns = b.columns = ['tweet_index', 'label', 'text']
        if not set(a.label).issubset({'0', '1'}) or not set(b.label).issubset({'0', '1', '2', '3'}):
            raise ValueError('Unrecognized SemEval label definition')
        joined = a.merge(b, on='tweet_index', suffixes=('_a', '_b'), validate='one_to_one', how='outer', indicator=True)
        # Upstream training row 2305 represents line separators differently in A and B.
        compatible = joined.apply(lambda r: ' '.join(r.text_a.replace('|', ' ').split()) ==
                                  ' '.join(r.text_b.replace('\\n', ' ').split()), axis=1)
        if not joined._merge.eq('both').all() or not (joined.text_a.eq(joined.text_b) |
                ((joined.tweet_index == '2305') & (split == 'train') & compatible)).all():
            raise ValueError('Task A/B identifier or target-text mismatch')
        if not joined.label_a.astype(int).eq(joined.label_b.astype(int).gt(0).astype(int)).all():
            raise ValueError('Task A/B annotation mismatch')
        frames.append(pd.DataFrame({'id': split + ':' + joined.tweet_index,
                                    'tweet_index': joined.tweet_index, 'text': joined.text_a,
                                    'label': joined.label_a.astype(int), 'irony_type': joined.label_b,
                                    'partition': split}))
    return standardize(pd.concat(frames, ignore_index=True), 'semeval2018', 'Twitter'), {
        'annotation': 'Manual fine-grained irony annotation',
        'task_b_codes': {'0': 'non-ironic', '1': 'verbal irony with polarity contrast',
                         '2': 'situational irony', '3': 'other verbal irony'},
        'join': 'One-to-one tweet index within partition and binary-label agreement verified. Training index 2305 has | versus literal backslash-n line separators; normalized wording verified, Task A text retained. All other texts match exactly.',
        'limitations': 'Hashtag collection bias; subtype categories are not exact explicit/implicit incongruity labels.'}


def semeval2022():
    root = RAW / 'semeval2022'
    train = pd.read_csv(root / 'train/train.En.csv').rename(columns={
        'tweet': 'text', 'sarcastic': 'label', 'rephrase': 'rephrasing', 'Unnamed: 0': 'original_index'})
    categories = ['sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question']
    train['irony_type'] = train.apply(lambda r: json.dumps([c for c in categories if r[c] == 1]), axis=1)
    train['id'] = 'train:' + train.original_index.astype(str)
    train['partition'] = 'train'
    test = pd.read_csv(root / 'test/task_A_En_test.csv').rename(columns={'sarcastic': 'label'})
    test['id'] = ['test:' + str(i) for i in range(len(test))]
    test['partition'] = 'test'
    return standardize(pd.concat([train, test], ignore_index=True), 'semeval2022', 'Twitter'), {
        'annotation': 'Author-provided intended sarcasm, expert figurative categories',
        'limitations': 'Rephrasings and categories available in training release; not added as independent examples. Test has binary labels only.'}


def figlang(platform):
    frames = []
    for partition, suffix in [('train', 'training'), ('test', 'testing')]:
        path = RAW / 'figlang2020' / platform / f'sarcasm_detection_shared_task_{platform}_{suffix}.jsonl'
        data = pd.read_json(path, lines=True).rename(columns={'response': 'text'})
        if not set(data.label).issubset({'SARCASM', 'NOT_SARCASM'}):
            raise ValueError('Missing/unrecognized FigLang gold labels')
        data['label'] = data.label.map({'SARCASM': 1, 'NOT_SARCASM': 0})
        data['context'] = data.context.map(lambda v: json.dumps(v, ensure_ascii=False))
        data['id'] = partition + ':' + (data.id.astype(str) if 'id' in data else pd.Series(range(len(data))).astype(str))
        data['partition'] = partition
        frames.append(data)
    return standardize(pd.concat(frames, ignore_index=True), 'figlang2020', platform.title()), {
        'annotation': 'Shared-task sarcasm labels; see task report for platform-specific collection and annotation',
        'limitations': 'No thread/author IDs. Shared context messages proxy conversation links; cannot guarantee identification of all original threads.',
        'test_gold': 'Both current upstream testing files contain 1800 gold labels.'}


def nikesh66():
    data = pd.read_csv(RAW / 'nikesh66/sarcasm_tweets.csv').rename(columns={'Tweet': 'text', 'Sarcasm (yes/no)': 'label'})
    if not set(data.label).issubset({'yes', 'no'}):
        raise ValueError('Unexpected nikesh66 label')
    data['label'] = data.label.map({'yes': 1, 'no': 0})
    data['id'] = ['row:' + str(i) for i in range(len(data))]
    data['partition'] = 'train'
    # Delexicalize observed subject phrases to avoid subject substitutions crossing splits.
    subjects = r'\b(?:my new phone|this movie|this book|the (?:meal|game|concert|workshop|conference)|musicians|vegetarians|artists|writers|athletes|doctors|engineers)\b'
    data['template_id'] = data.text.str.casefold().str.replace(subjects, '<subject>', regex=True)
    return standardize(data, 'nikesh66', 'Unverified'), {
        'annotation': 'Undocumented; yes/no labels interpreted using actual sarcasm column, not card slang description',
        'limitations': 'Exploratory only. No license or collection/annotation method in card; card incorrectly describes slang. Template subjects inferred from observed corpus, no claim of verified synthetic generation.',
        'template_policy': 'Mask 15 observed subject phrases; retain entire resulting template within a split.'}


def sample_sequences(path, limit):
    """Reservoir sampling of responses, bounded memory; iterate the entire label partition."""
    rng = random.Random(SEED)
    sample, total, positives = [], 0, 0
    with bz2.open(path, 'rt', encoding='utf-8') as f:
        for ancestors, responses, labels in csv.reader(f, delimiter='|'):
            responses, labels = responses.split(), labels.split()
            if len(responses) != len(labels):
                raise ValueError('SARC response/label alignment mismatch')
            for response, label in zip(responses, labels):
                item = {'comment_id': response, 'ancestor_ids': ancestors.split(), 'label': int(label)}
                positives += int(label)
                total += 1
                if not limit or len(sample) < limit:
                    sample.append(item)
                else:
                    index = rng.randrange(total)
                    if index < limit:
                        sample[index] = item
    return sample, {'count': total, 'positive': positives, 'negative': total - positives}


def sarc(sample_size, test_size):
    import ijson
    root = RAW / 'sarc/main'
    rows, totals = [], {}
    for partition, size in [('train', sample_size), ('test', test_size)]:
        selected, total = sample_sequences(root / f'{partition}-balanced.csv.bz2', size)
        for row in selected:
            row['partition'] = partition
        rows += selected
        totals[partition] = total
    needed = {r['comment_id'] for r in rows} | {a for r in rows for a in r['ancestor_ids']}
    comments = {}
    print(f'SARC: streaming comment archive for {len(needed)} IDs...', flush=True)
    with bz2.open(root / 'comments.json.bz2', 'rb') as f:
        for ident, value in ijson.kvitems(f, ''):
            if ident in needed:
                comments[ident] = value
    missing = needed - comments.keys()
    if missing:
        raise ValueError(f'SARC missing {len(missing)} required comment records')
    records = []
    for r in rows:
        comment = comments[r['comment_id']]
        context = [comments[a]['text'] for a in r['ancestor_ids']]
        records.append({**comment, 'id': r['partition'] + ':' + r['comment_id'], 'text': comment['text'],
                        'label': r['label'], 'comment_id': r['comment_id'], 'partition': r['partition'],
                        'conversation_id': r['ancestor_ids'][0], 'ancestor_ids': json.dumps(r['ancestor_ids']),
                        'parent_comment': context[-1] if context else '', 'context': json.dumps(context, ensure_ascii=False)})
    return standardize(pd.DataFrame(records), 'sarc', 'Reddit'), {
        'annotation': 'Author self-annotation using /s', 'release': 'Official SARC 2.0 main balanced',
        'full_partition_counts': totals, 'requested_training_pool': sample_size, 'requested_test_sample': test_size,
        'limitations': 'Sampled official test, not full published benchmark; balanced corpus does not represent natural prevalence. Non-sarcastic labels may include unmarked sarcasm.',
        'sampling': 'Seed 42 response reservoir separately per official partition; fixed test sample independent of training pool size. Sample before cleaning; final sizes may be smaller.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'semeval2018', 'semeval2022', 'figlang2020', 'sarc', 'nikesh66'], default='all')
    parser.add_argument('--sarc-sample-size', type=int, default=10000, help='Official training pool responses; 0 loads all (memory intensive)')
    parser.add_argument('--sarc-test-size', type=int, default=2000, help='Fixed official test sample; 0 for all')
    args = parser.parse_args()
    if args.sarc_sample_size < 0 or args.sarc_test_size < 0:
        parser.error('Sample sizes must be nonnegative')
    for dataset in ['semeval2018', 'semeval2022', 'figlang2020', 'sarc', 'nikesh66']:
        if args.dataset not in ['all', dataset]:
            continue
        if dataset == 'figlang2020':
            for platform in ['twitter', 'reddit']:
                frame, meta = figlang(platform)
                prepare(frame, f'figlang2020/{platform}', meta)
        else:
            frame, meta = sarc(args.sarc_sample_size, args.sarc_test_size) if dataset == 'sarc' else globals()[dataset]()
            prepare(frame, dataset, meta)


if __name__ == '__main__':
    main()
