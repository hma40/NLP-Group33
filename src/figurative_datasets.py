"""Academic humor and idiomaticity extensions; never relabel these as sarcasm."""
import json
import zipfile

import pandas as pd

try:
    from .data_common import RAW, PROCESSED, standardize
    from .dataset_catalog import TASKS
except ImportError:
    from data_common import RAW, PROCESSED, standardize
    from dataset_catalog import TASKS


def add_task(frame, name, platform):
    task, positive, negative = TASKS[name]
    frame['task'] = task
    frame['positive_label'] = positive
    frame['negative_label'] = negative
    # Sarcasm collection hashtags are not gold labels for humor or idiomaticity.
    return standardize(frame, name, platform, strip_label_markers=False)


def hahackathon():
    frames = []
    with zipfile.ZipFile(RAW / 'hahackathon/hahackathon_data.zip') as archive:
        for split in ['train', 'dev', 'test']:
            frame = pd.read_csv(archive.open(f'hahackathon data/{split}.csv'))
            frame = frame.rename(columns={'is_humor': 'label'})
            frame['original_id'] = frame.id
            frame['id'] = split + ':' + frame.id.astype(str)
            frame['partition'] = split
            frames.append(frame)
    return add_task(pd.concat(frames, ignore_index=True), 'hahackathon', 'Twitter / short jokes'), {
        'annotation': '20 human annotators per text, ages 18-70; aggregated humor judgments',
        'task': TASKS['hahackathon'][0], 'positive_label': TASKS['hahackathon'][1],
        'preserve_official_dev': True,
        'limitations': 'Humor is subjective and may overlap sarcasm. Mixed Twitter/short-joke domains; per-row origin unavailable. Ratings and controversy are label-related metadata, not model inputs. No explicit license in organizer ZIP/resource listing.',
        'split_policy': 'Preserve official train/dev/test roles; exclude train links to holdouts, dev links to test if found.',
        'cleaning': 'Whitespace trimming only; sarcasm markers retained because they are not this task label.'}


def idiom_rows(frame, partition, prefix):
    frame = frame.loc[frame.Language.eq('EN')].copy()
    if not set(frame.Label.unique()).issubset({0, 1}):
        raise ValueError('Unexpected official idiomaticity labels')
    frame['source_label'] = frame.Label.astype(int)
    # Official Table 2: 0=idiomatic, 1=non-idiomatic, including proper names.
    frame['label'] = 1 - frame.source_label
    frame['text'] = frame.Target
    frame['id'] = prefix + ':' + frame.DataID.astype(str)
    frame['partition'] = partition
    frame['original_partition'] = prefix
    frame['mwe'] = frame.MWE
    frame['template_id'] = 'mwe:' + frame.MWE.str.casefold().str.strip()
    frame['context'] = frame.Previous.fillna('').map(lambda text: json.dumps([text] if text.strip() else [], ensure_ascii=False))
    frame['following_context'] = frame.Next.fillna('')
    return frame.drop(columns=['Label', 'Target', 'Previous', 'Next'])


def semeval2022_idioms():
    root = RAW / 'semeval2022_idioms/SubTaskA'
    train = pd.read_csv(root / 'Data/train_zero_shot.csv')
    dev = pd.read_csv(root / 'Data/dev.csv')
    gold = pd.read_csv(root / 'Data/dev_gold.csv')
    dev = dev.merge(gold, on=['ID', 'Language'], validate='one_to_one', how='outer', indicator=True)
    if not dev._merge.eq('both').all():
        raise ValueError('Unmatched official dev IDs/gold labels')
    dev = dev.drop(columns='_merge')
    frames = [idiom_rows(train, 'train', 'official_train_zero_shot'),
              idiom_rows(dev, 'train', 'official_dev')]
    # The official test has no published gold in this release. Keep it explicitly unlabeled.
    test = pd.read_csv(root / 'TestData/test.csv')
    test = test.loc[test.Language.eq('EN')].copy()
    out = PROCESSED / 'semeval2022_idioms'
    out.mkdir(parents=True, exist_ok=True)
    test.to_csv(out / 'unlabeled_official_test.csv', index=False)
    return add_task(pd.concat(frames, ignore_index=True), 'semeval2022_idioms', 'Web text'), {
        'annotation': 'Human annotation of potentially idiomatic multiword expressions in context',
        'task': TASKS['semeval2022_idioms'][0], 'positive_label': TASKS['semeval2022_idioms'][1],
        'limitations': 'Web-domain auxiliary task, not social-media sarcasm. Labels describe the specified MWE usage, not every figurative expression in the sentence. Official test gold unavailable; local evaluation is not official shared-task performance.',
        'source_label_mapping': 'Official 0=idiomatic -> project 1; official 1=non-idiomatic -> project 0.',
        'split_policy': 'English zero-shot training plus ID-joined official dev gold; new seed-42 approximately 80/10/10 MWE-disjoint local splits. Exclude one-shot and trial variants to avoid duplicate releases.',
        'unlabeled_official_test_count': len(test),
        'context_policy': 'context contains Previous only; following_context holds Next separately.',
        'cleaning': 'Whitespace trimming only; no sarcasm-marker stripping.'}
