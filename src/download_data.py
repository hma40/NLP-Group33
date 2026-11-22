"""Fetch verified source files locally; no data are uploaded or redistributed."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
try:
    from .dataset_catalog import PREPARATION_CHOICES
except ImportError:
    from dataset_catalog import PREPARATION_CHOICES


def sha256(path):
    checksum = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='all', choices=['all'] + PREPARATION_CHOICES)
    parser.add_argument('--verify-only', action='store_true', help='Check local files without network access')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'config/dataset_sources.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        if args.dataset not in ['all', item['dataset']]:
            continue
        dest = ROOT / item['path']
        if dest.exists() and sha256(dest) == item['sha256']:
            print('Verified:', item['path'], flush=True)
            continue
        if args.verify_only:
            raise ValueError(f'Missing or changed source: {dest}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + '.part')
        print('Downloading:', item['url'], flush=True)
        with requests.get(item['url'], stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temp.open('wb') as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        if sha256(temp) != item['sha256']:
            raise ValueError(f'Source checksum mismatch: {temp}; do not use changed release without reviewing it.')
        for attempt in range(6):
            try:
                temp.replace(dest)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.5)


if __name__ == '__main__':
    main()
