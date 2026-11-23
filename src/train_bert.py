"""Fine-tune a RoBERTa classifier for binary irony/sarcasm detection.

Three modes, run in order:

    --search     grid search on dev, then freeze selected_config.json
    --final      retrain the frozen config with several seeds (dev only)
    --eval-test  score each final checkpoint on test once

Only the `text` column is used as model input. The default modes never read test.csv.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import logging as hf_logging
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 'cardiffnlp/twitter-roberta-base'
MAX_LENGTH = 128
WARMUP_FRACTION = 0.1
WEIGHT_DECAY = 0.01
SEARCH_SEED = 42


# ---------------------------------------------------------------- data

def normalize(texts):
    """Cardiff Twitter convention, applied per whitespace token: '@name' -> '@user', 'http...' -> 'http'.

    Everything else (emoji, punctuation, case) is kept. Returns new texts and per-row counts.
    """
    counts = {'mention_rows': 0, 'url_rows': 0}
    out = []
    for text in texts:
        tokens = text.split(' ')
        mention = [t.startswith('@') and len(t) > 1 for t in tokens]
        url = [t.startswith('http') for t in tokens]
        counts['mention_rows'] += any(m and t != '@user' for m, t in zip(mention, tokens))
        counts['url_rows'] += any(u and t != 'http' for u, t in zip(url, tokens))
        tokens = ['@user' if m else 'http' if u else t for t, m, u in zip(tokens, mention, url)]
        out.append(' '.join(tokens))
    return out, counts


def load_split(data_dir, split, limit=None):
    frame = pd.read_csv(data_dir / f'{split}.csv', keep_default_na=False)
    if limit:
        frame = frame.head(limit)
    frame['label'] = frame['label'].astype(int)
    return frame


class Encoded(torch.utils.data.Dataset):
    """Tokenized `text` and `label`. No other column reaches the model."""

    def __init__(self, frame, tokenizer, name):
        texts, counts = normalize(frame['text'].astype(str).tolist())
        full_lengths = [len(ids) for ids in tokenizer(texts)['input_ids']]
        truncated = sum(n > MAX_LENGTH for n in full_lengths)
        print(f'[{name}] rows={len(texts)} mentions_replaced={counts["mention_rows"]} '
              f'urls_replaced={counts["url_rows"]} truncated={truncated} '
              f'max_tokens={max(full_lengths)}', flush=True)
        self.stats = {**counts, 'truncated': truncated, 'rows': len(texts)}
        self.encodings = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
        self.labels = frame['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item['labels'] = self.labels[i]
        return item


# ---------------------------------------------------------------- training

def pick_device(requested=None):
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}', flush=True)
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(labels, preds):
    return {'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1_positive': f1_score(labels, preds, zero_division=0),
            'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
            'accuracy': accuracy_score(labels, preds)}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        logits = model(**batch).logits.float()
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    prob = torch.cat(probs).numpy()
    return (prob >= 0.5).astype(int), prob


def better(a, b):
    """Selection rule: positive-class F1, ties broken by macro F1."""
    return b is None or (a['f1_positive'], a['macro_f1']) > (b['f1_positive'], b['macro_f1'])


def train_run(args, tokenizer, train_set, dev_set, lr, batch_size, seed, device, save_dir=None):
    """Train for up to args.epochs; return per-epoch log, best-epoch dev predictions and timing.

    If save_dir is given, the best epoch's weights are saved there (overwritten on improvement).
    """
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    collate = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    dev_loader = DataLoader(dev_set, batch_size=64, collate_fn=collate)

    no_decay = ('bias', 'LayerNorm.weight', 'layer_norm.weight')
    groups = [{'params': [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
               'weight_decay': WEIGHT_DECAY},
              {'params': [p for n, p in model.named_parameters() if n.endswith(no_decay)],
               'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * total_steps), total_steps)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    log, best, start = [], None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        preds, prob = predict(model, dev_loader, device)
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), **metrics(dev_set.labels, preds)}
        log.append(row)
        print(f'  lr={lr} bs={batch_size} seed={seed} epoch={epoch} loss={row["train_loss"]:.4f} '
              f'dev_f1_pos={row["f1_positive"]:.4f} dev_macro_f1={row["macro_f1"]:.4f}', flush=True)
        if better(row, best):
            best = {**row, 'preds': preds, 'prob': prob}
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
    n_params = sum(p.numel() for p in model.parameters())
    return log, best, time.time() - start, n_params


# ---------------------------------------------------------------- modes

def run_search(args, paths, tokenizer, train_set, dev_set, device):
    if paths['config'].exists() and not args.force:
        raise SystemExit(f'{paths["config"]} already exists; the config is frozen. '
                         'Pass --force to rerun the search and overwrite it.')
    rows, start = [], time.time()
    for lr, batch_size in product(args.lrs, args.batch_sizes):
        log, _, _, _ = train_run(args, tokenizer, train_set, dev_set, lr, batch_size, SEARCH_SEED, device)
        rows += [{'lr': lr, 'batch_size': batch_size, **row} for row in log]
    search = pd.DataFrame(rows)
    search.to_csv(paths['out'] / 'search.csv', index=False)
    ranked = search.sort_values(['f1_positive', 'macro_f1'], ascending=False, kind='stable')
    top = ranked.iloc[0]
    config = {'dataset': args.dataset, 'model': args.model,
              'learning_rate': float(top.lr), 'batch_size': int(top.batch_size),
              'max_epochs': args.epochs, 'best_search_epoch': int(top.epoch),
              'epoch_selection': 'best dev epoch per run (f1_positive, then macro_f1)',
              'warmup_fraction': WARMUP_FRACTION, 'weight_decay': WEIGHT_DECAY,
              'max_length': MAX_LENGTH, 'search_seed': SEARCH_SEED,
              'search_dev_f1_positive': float(top.f1_positive), 'search_dev_macro_f1': float(top.macro_f1),
              'limit': args.limit, 'search_seconds': round(time.time() - start, 1),
              'frozen_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    paths['config'].write_text(json.dumps(config, indent=2))
    print(f'Selected lr={config["learning_rate"]} batch_size={config["batch_size"]} '
          f'(epoch {config["best_search_epoch"]}, dev F1 {config["search_dev_f1_positive"]:.4f}). '
          f'Frozen in {paths["config"]}', flush=True)


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--search', action='store_true', help='grid search on dev and freeze the config')
    mode.add_argument('--final', action='store_true', help='retrain the frozen config with --seeds')
    mode.add_argument('--eval-test', action='store_true', help='evaluate final checkpoints on test once')
    p.add_argument('--dataset', default='semeval2018')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--lrs', type=float, nargs='+', default=[1e-5, 2e-5, 3e-5])
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 32])
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--limit', type=int, help='use only the first N train and dev rows (debugging)')
    p.add_argument('--device', help='override device selection, e.g. cpu')
    p.add_argument('--force', action='store_true', help='allow overwriting a frozen config')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'Data' / 'Processed')
    p.add_argument('--results-dir', type=Path, default=ROOT / 'results' / 'bert')
    p.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints' / 'bert')
    return p.parse_args(argv)


def output_paths(args):
    name = args.model.rstrip('/').split('/')[-1]
    if args.limit:
        name += f'-limit{args.limit}'  # keep debug runs away from real results
    out = args.results_dir / args.dataset / name
    out.mkdir(parents=True, exist_ok=True)
    return {'out': out, 'config': out / 'selected_config.json',
            'checkpoints': args.checkpoint_dir / args.dataset / name}


def main(argv=None):
    args = parse_args(argv)
    paths = output_paths(args)
    data_dir = args.data_dir / args.dataset
    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_set = Encoded(load_split(data_dir, 'train', args.limit), tokenizer, 'train')
    dev_set = Encoded(load_split(data_dir, 'dev', args.limit), tokenizer, 'dev')
    if args.search:
        run_search(args, paths, tokenizer, train_set, dev_set, device)


if __name__ == '__main__':
    main()
