"""End-to-end and leakage checks for the transformer baseline (CPU, tiny model)."""
import importlib.util
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None
if HAS_TORCH:
    from transformers import AutoTokenizer
    from src import train_bert

TINY = 'hf-internal-testing/tiny-random-roberta'
TEST_TEMP = Path(__file__).resolve().parents[1] / '.cache' / 'tests'
PRED_COLUMNS = ['id', 'label', 'pred', 'prob_positive']


def synthetic_split(split, n):
    """Small frame with the processed-data schema."""
    rows = []
    for i in range(n):
        label = i % 2
        rows.append({'id': f'{split}:{i}',
                     'text': (f'@friend{i} oh great, another monday {i} 🙃 http://t.co/x{i}' if label
                              else f'Lunch with family number {i} today'),
                     'label': label, 'irony_type': (1 + i % 3) if label else 0, 'context': '[]',
                     'source': 'fixture', 'platform': 'Twitter', 'original_text': 'unused',
                     'group_id': f'g{split}{i}'})
    return pd.DataFrame(rows)


@unittest.skipUnless(HAS_TORCH, 'torch/transformers not installed (pip install -r requirements-bert.txt)')
class TrainBertTests(unittest.TestCase):
    def setUp(self):
        self.temp = TEST_TEMP / uuid4().hex
        self.data = self.temp / 'data' / 'fixture'
        self.data.mkdir(parents=True)
        self.train = synthetic_split('train', 40)
        self.dev = synthetic_split('dev', 40)
        self.test = synthetic_split('test', 20)
        self.train.to_csv(self.data / 'train.csv', index=False)
        self.dev.to_csv(self.data / 'dev.csv', index=False)
        # test.csv is written only right before --eval-test: earlier modes must not need it.
        self.out = self.temp / 'results' / 'fixture' / 'tiny-random-roberta-limit32'

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def run_cli(self, *mode):
        train_bert.main([*mode, '--dataset', 'fixture', '--model', TINY, '--limit', '32',
                         '--epochs', '1', '--lrs', '1e-5', '--batch-sizes', '16', '--device', 'cpu',
                         '--data-dir', str(self.temp / 'data'),
                         '--results-dir', str(self.temp / 'results'),
                         '--checkpoint-dir', str(self.temp / 'checkpoints')])

    def test_full_pipeline_writes_expected_outputs(self):
        self.run_cli('--search')
        self.run_cli('--final', '--seeds', '42')
        for name in ['search.csv', 'selected_config.json', 'train_log_seed42.csv',
                     'subtype_recall_dev.csv', 'summary.json']:
            self.assertTrue((self.out / name).exists(), name)
        dev_pred = pd.read_csv(self.out / 'predictions_dev_seed42.csv', keep_default_na=False)
        self.assertEqual(dev_pred.columns.tolist(), PRED_COLUMNS)
        self.assertEqual(len(dev_pred), 32)
        self.assertEqual(dev_pred.id.tolist(), self.dev.id.head(32).tolist())
        self.assertTrue(dev_pred.prob_positive.between(0, 1).all())

        self.test.to_csv(self.data / 'test.csv', index=False)
        self.run_cli('--eval-test', '--seeds', '42')
        test_pred = pd.read_csv(self.out / 'predictions_test_seed42.csv', keep_default_na=False)
        self.assertEqual(test_pred.columns.tolist(), PRED_COLUMNS)
        self.assertEqual(len(test_pred), len(self.test))
        summary = json.loads((self.out / 'summary.json').read_text())
        self.assertIn('test', summary['seeds']['42'])
        self.assertIn('test_evaluated_at', summary['seeds']['42'])
        # A second test evaluation of the same seed is refused.
        with self.assertRaises(SystemExit):
            self.run_cli('--eval-test', '--seeds', '42')

    def test_eval_test_requires_frozen_config(self):
        self.test.to_csv(self.data / 'test.csv', index=False)
        with self.assertRaises(SystemExit) as raised:
            self.run_cli('--eval-test', '--seeds', '42')
        self.assertIn('selected_config.json', str(raised.exception))
        self.assertFalse((self.out / 'predictions_test_seed42.csv').exists())

    def test_irony_type_and_metadata_do_not_reach_model(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY)
        original = train_bert.Encoded(self.dev, tokenizer, 'original')
        altered_frame = self.dev.copy()
        altered_frame['irony_type'] = 3 - altered_frame['irony_type']
        altered_frame['context'] = '["ironic sarcastic context"]'
        altered_frame['original_text'] = 'SARCASM'
        altered = train_bert.Encoded(altered_frame, tokenizer, 'altered')
        self.assertEqual(original.encodings['input_ids'], altered.encodings['input_ids'])
        self.assertEqual(set(original[0]), {'input_ids', 'attention_mask', 'labels'})

    def test_normalization_replaces_mentions_and_urls_only(self):
        texts, counts = train_bert.normalize(['@bob Love it 😍 https://t.co/a!', 'Mail me@x.com', 'plain'])
        self.assertEqual(texts, ['@user Love it 😍 http', 'Mail me@x.com', 'plain'])
        self.assertEqual(counts, {'mention_rows': 1, 'url_rows': 1})


if __name__ == '__main__':
    unittest.main()
