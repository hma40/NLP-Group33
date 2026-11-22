"""Regression checks for the data decisions that can contaminate evaluation."""
import bz2
import json
import tempfile
import unittest
from contextlib import contextmanager
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_common import clean, key, leakage_groups, prepare, split_pool, standardize, validate
from src.preprocess import sample_sequences
from src.figurative_datasets import idiom_rows, add_task

TEST_TEMP = Path(__file__).resolve().parents[1] / '.cache' / 'tests'
TEST_TEMP.mkdir(parents=True, exist_ok=True)


@contextmanager
def test_directory():
    # Normal inherited Windows ACLs; tempfile's restrictive ACLs conflict with this sandbox.
    path = TEST_TEMP / uuid4().hex
    path.mkdir()
    yield path


class PreparationTests(unittest.TestCase):
    def test_idiom_labels_are_inverted_and_context_directions_preserved(self):
        data = pd.DataFrame({'Language': ['EN', 'EN', 'PT'], 'Label': [0, 1, 0],
                             'Target': ['An idiomatic use', 'A literal use', 'Not English'],
                             'DataID': ['a', 'b', 'c'], 'MWE': ['old hat'] * 3,
                             'Previous': ['Earlier sentence', None, None],
                             'Next': ['Following sentence', None, None]})
        mapped = idiom_rows(data, 'train', 'fixture')
        self.assertEqual(mapped.label.tolist(), [1, 0])
        self.assertEqual(mapped.source_label.tolist(), [0, 1])
        self.assertEqual(json.loads(mapped.context.iloc[0]), ['Earlier sentence'])
        self.assertEqual(mapped.following_context.iloc[0], 'Following sentence')
        frame = add_task(mapped, 'semeval2022_idioms', 'Web text')
        groups, _ = leakage_groups(frame)
        self.assertEqual(groups[0], groups[1])

    def test_humor_does_not_strip_sarcasm_markers(self):
        frame = pd.DataFrame({'id': ['a'], 'text': ['WOW!!! #sarcasm 😂'], 'label': [1]})
        standardized = add_task(frame, 'hahackathon', 'Twitter / short jokes')
        self.assertEqual(standardized.text.iloc[0], frame.text.iloc[0])
        self.assertEqual(standardized.positive_label.iloc[0], 'humorous')
        self.assertFalse(standardized.attrs['strip_label_markers'])

    def test_official_development_is_not_resplit(self):
        records = [{'id': str(i), 'text': f'Independent text {chr(97+i)}', 'label': i % 2,
                    'partition': 'train' if i < 16 else 'dev' if i < 18 else 'test'} for i in range(20)]
        frame = standardize(pd.DataFrame(records), 'fixture', 'test')
        with test_directory() as temp:
            with patch('src.data_common.PROCESSED', temp / 'processed'), patch('src.data_common.STATS', temp / 'stats'):
                report = prepare(frame, 'fixture', {'preserve_official_dev': True})
            actual = pd.read_csv(temp / 'processed/fixture/dev.csv', dtype={'id': str})
            self.assertEqual(set(actual.id), {'16', '17'})
            self.assertEqual(report['splits']['test']['count'], 2)

    def test_minimal_cleaning(self):
        self.assertEqual(clean(' WOW!!! Soooo good 😂 #sarcasm /s '), 'WOW!!! Soooo good 😂')
        self.assertEqual(clean('I do not like this!'), 'I do not like this!')
        self.assertNotEqual(key('😂'), key('😢'))

    def test_transitive_context_rephrasing_and_author_groups(self):
        data = pd.DataFrame([
            {'id': 'a', 'text': 'first unique statement here', 'label': 0, 'rephrasing': 'second linked statement'},
            {'id': 'b', 'text': 'second linked statement', 'label': 1, 'author': 'person'},
            {'id': 'c', 'text': 'third unrelated wording', 'label': 0, 'author': 'person'},
            {'id': 'd', 'text': 'fourth separate response', 'label': 1, 'context': json.dumps(['first unique statement here'])}])
        groups, _ = leakage_groups(standardize(data, 'fixture', 'test'))
        self.assertEqual(len(set(groups)), 1)

    def test_stratified_split_is_repeatable(self):
        frame = pd.DataFrame({'id': range(100), 'label': [0, 1] * 50, 'group_id': range(100)})
        train, dev = split_pool(frame, .2)
        train_again, dev_again = split_pool(frame, .2)
        self.assertEqual((len(train), len(dev), int(dev.label.sum())), (80, 20, 10))
        self.assertEqual(dev.id.tolist(), dev_again.id.tolist())

    def test_official_test_kept_and_duplicate_context_bridge_removed(self):
        records = [{'id': str(i), 'text': f'unique {chr(97+i)} standalone sample {chr(65+i)}',
                    'label': i % 2, 'partition': 'train'} for i in range(20)]
        records += [
            {'id': 'dup1', 'text': 'duplicated target wording', 'label': 0, 'partition': 'train', 'context': json.dumps(['bridge context sentence long enough'])},
            {'id': 'dup2', 'text': 'duplicated target wording', 'label': 0, 'partition': 'train', 'context': json.dumps(['official context sentence long enough'])},
            {'id': 'linked', 'text': 'freshly authored reply', 'label': 1, 'partition': 'train', 'context': json.dumps(['bridge context sentence long enough'])},
            {'id': 'test', 'text': 'official held out target', 'label': 1, 'partition': 'test', 'context': json.dumps(['official context sentence long enough'])}]
        frame = standardize(pd.DataFrame(records), 'fixture', 'test')
        with test_directory() as temp:
            root = Path(temp)
            with patch('src.data_common.PROCESSED', root / 'processed'), patch('src.data_common.STATS', root / 'stats'):
                report = prepare(frame, 'fixture')
            test = pd.read_csv(root / 'processed/fixture/test.csv')
            self.assertEqual(test.id.tolist(), ['test'])
            self.assertEqual(report['removed_train_rows_linked_to_official_test'], 2)

    def test_reservoir_sampling(self):
        with test_directory() as temp:
            path = Path(temp) / 'train.csv.bz2'
            with bz2.open(path, 'wt') as file:
                for i in range(100):
                    file.write(f'root parent|r{i}|{i % 2}\n')
            a, counts = sample_sequences(path, 10)
            b, _ = sample_sequences(path, 10)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 10)
            self.assertEqual(counts, {'count': 100, 'positive': 50, 'negative': 50})


if __name__ == '__main__':
    unittest.main()
