"""Aggregate statistics and charts only; notebooks call these same functions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .data_common import PROCESSED, ROOT, STATS, key
    from .dataset_catalog import DATASETS, TASKS
except ImportError:
    from data_common import PROCESSED, ROOT, STATS, key
    from dataset_catalog import DATASETS, TASKS


def explore(name):
    audit = json.loads((STATS / (name.replace('/', '_') + '.json')).read_text(encoding='utf-8'))
    data = pd.concat([pd.read_csv(PROCESSED / name / f'{s}.csv', keep_default_na=False).assign(split=s)
                      for s in ['train', 'dev', 'test']], ignore_index=True)
    figures = ROOT / 'results/figures'
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    data.groupby(['split', 'label']).size().unstack(fill_value=0).reindex(['train', 'dev', 'test']).plot.bar(ax=axes[0])
    axes[0].set_title('Class counts by split')
    axes[0].set_ylabel('Examples')
    data.text.str.len().hist(bins=35, ax=axes[1])
    axes[1].set_title('Target length (characters)')
    axes[1].set_xlabel('Characters')
    data.split.value_counts(normalize=True).reindex(['train', 'dev', 'test']).mul(100).plot.bar(ax=axes[2])
    axes[2].set_title('Split proportions')
    axes[2].set_ylabel('Percent of prepared examples')
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(figures / f'{name.replace("/", "_")}.png', dpi=140)
    plt.close(fig)
    lengths = data.text.str.len()
    histogram = pd.cut(lengths, bins=[-1, 20, 50, 100, 150, 250, 500, 1000, float('inf')]).value_counts(sort=False)
    histogram.rename_axis('characters').to_csv(STATS / f'{name.replace("/", "_")}_length_histogram.csv', header=['count'])
    return audit


def comparison():
    rows = []
    text_sets = {}
    exact_sets = {}
    split_sets = {}
    for name in DATASETS:
        audit = explore(name)
        meta = audit['metadata']
        first = pd.read_csv(PROCESSED / name / 'train.csv', nrows=1)
        text_sets[name] = set()
        exact_sets[name] = set()
        for split in ['train', 'dev', 'test']:
            texts = pd.read_csv(PROCESSED / name / f'{split}.csv', usecols=['text'], keep_default_na=False).text
            text_sets[name].update(texts.map(key))
            exact_sets[name].update(texts)
            split_sets[name, split] = set(texts.map(key))
        rows.append({'dataset': name, 'platform': first.platform.iloc[0], 'annotation': meta['annotation'],
                     'task': TASKS[name][0], 'positive_label': TASKS[name][1], 'negative_label': TASKS[name][2],
                     'original_source_size': sum(p['count'] for p in meta['full_partition_counts'].values())
                         if 'full_partition_counts' in meta else audit['input']['count'],
                     'original_loaded_size': audit['input']['count'], 'processed_size': audit['processed']['count'],
                     **{s: audit['splits'][s]['count'] for s in ['train', 'dev', 'test']},
                     'positive_percent': audit['processed']['positive_percent'],
                     'context_count': audit['processed']['context_count'],
                     'subtypes': json.dumps(audit['processed']['irony_categories']),
                     'limitations': meta['limitations']})
    table = pd.DataFrame(rows)
    table.to_csv(STATS / 'comparison.csv', index=False)
    overlap = [{'dataset_a': a, 'dataset_b': b, 'shared_normalized_targets': len(text_sets[a] & text_sets[b]),
                'shared_exact_targets': len(exact_sets[a] & exact_sets[b])}
               for a in text_sets for b in text_sets if a < b]
    pd.DataFrame(overlap).to_csv(STATS / 'cross_dataset_overlap.csv', index=False)
    detailed = [{'dataset_a': a, 'split_a': sa, 'dataset_b': b, 'split_b': sb,
                 'shared_normalized_targets': len(split_sets[a, sa] & split_sets[b, sb])}
                for a, sa in split_sets for b, sb in split_sets if a < b]
    pd.DataFrame(detailed).to_csv(STATS / 'cross_dataset_split_overlap.csv', index=False)
    lines = ['# Observed preparation results', '',
             'Generated from audit files. SARC rows describe the default selected sample, not the whole corpus.', '',
             'Tasks stay separate: positive means irony/sarcasm, humor, or idiomatic usage according to the dataset. Idiom test is a local MWE-disjoint holdout, not official gold test.', '',
             '| Dataset | Loaded | Processed | Train | Dev | Test | Positive % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['original_loaded_size']} | {row['processed_size']} | {row['train']} | {row['dev']} | {row['test']} | {row['positive_percent']:.2f} |")
    lines += ['', 'Use comparison.csv for annotation methods and limitations; per-dataset JSON for missing values, conflicts, lengths and removals.',
              'Cross-dataset overlap tables are diagnostics. Before transfer experiments, exclude target-evaluation overlaps on the source-training side using a separate derived split; retain the original benchmarks.', '']
    (STATS / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 4))
    table.set_index('dataset')[['original_loaded_size', 'processed_size']].plot.barh(ax=ax)
    ax.set_xlabel('Examples (SARC is a sample)')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/figures/dataset_comparison.png', dpi=140)
    plt.close(fig)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='all')
    args = parser.parse_args()
    if args.dataset == 'all':
        print(comparison().drop(columns=['annotation', 'limitations', 'subtypes']).to_string(index=False))
    else:
        print(json.dumps(explore(args.dataset), indent=2))
