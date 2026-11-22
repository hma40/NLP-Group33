"""Create nested SARC training subsets without changing development or test data."""
import argparse

import pandas as pd

try:
    from .data_common import PROCESSED, digest
except ImportError:
    from data_common import PROCESSED, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 5000])
    args = parser.parse_args()
    data = pd.read_csv(PROCESSED / 'sarc/train.csv', keep_default_na=False)
    if any(size <= 0 or size > len(data) for size in args.sizes):
        parser.error(f'Each size must be between 1 and {len(data)}. Prepare a larger base pool first if needed.')
    data['_priority'] = data.id.map(lambda value: digest('42:' + value))
    data = data.sort_values('_priority').drop(columns='_priority')
    out = PROCESSED / 'sarc/subsets'
    out.mkdir(exist_ok=True)
    for size in sorted(set(args.sizes)):
        subset = data.iloc[:size]
        subset.to_csv(out / f'train_{size}.csv', index=False)
        print(f'{size}: positive={subset.label.sum()}, negative={size - subset.label.sum()}')


if __name__ == '__main__':
    main()
