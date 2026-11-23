# NLP Group 33 — Detecting Verbal Irony in Social Media

Phase 1 prepares the original five irony/sarcasm datasets plus two auxiliary humor/idiom datasets. No classifiers or experimental scores are implemented here. SemEval 2018 remains the primary irony benchmark. Read the [November 21 expansion report](docs/FIGURATIVE_EXPANSION.md) for the response to proposal feedback, source selection, new task meanings and limitations.

## Windows quick start

Run these PowerShell commands from the repository root. Python 3.12 is validated. The current machine already has `.venv`; teammates create their own environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/download_data.py --dataset semeval2018
.\.venv\Scripts\python.exe src/preprocess.py --dataset semeval2018
```

SemEval 2018 is then ready at `Data/Processed/semeval2018/{train,dev,test}.csv` with 3,067 / 767 / 784 examples. For all seven datasets, including both FigLang platforms, a manageable official SARC sample, humor and English idioms:

```powershell
.\.venv\Scripts\python.exe src/download_data.py --dataset all
.\.venv\Scripts\python.exe src/preprocess.py --dataset all
.\.venv\Scripts\python.exe src/validate_data.py
.\.venv\Scripts\python.exe src/explore_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m jupyter notebook Notebooks
```

Read [dataset terms, provenance and methodology](docs/DATASETS.md) before downloading or distributing data. The SARC download is about 430 MiB, and its streaming scan can take several minutes. No GPU, torch, transformers or sentiment downloads are needed for preparation. Requirements retain the previously listed libraries; `ijson` adds streaming. `requirements-lock.txt` records the tested Windows Python 3.12 environment. Use it instead of the unpinned requirements when exact package reproduction on Windows is required. On macOS/Linux use `python` in your environment instead of the Windows executable path.

The existing `Data/Raw` and `Data/Processed` capitalization is preserved for compatibility with the repository. Scripts resolve paths with `pathlib` relative to their own source files, so their working directory does not control output locations. Raw and processed text are gitignored. Downloads are pinned by revision and checksum in [the source manifest](config/dataset_sources.json), and `download_data.py --verify-only` checks local bytes without network access.

## Model interface for teammates

Each directory has identical train/dev/test CSV names. The eight prepared portions are `semeval2018`, `semeval2022`, `figlang2020/twitter`, `figlang2020/reddit`, `sarc`, `nikesh66`, `hahackathon`, and `semeval2022_idioms`.

The added tasks remain independent: `hahackathon` uses 1 = humorous (7,996 / 1,000 / 1,000); `semeval2022_idioms` uses 1 = idiomatic use of the specified MWE (3,043 / 378 / 372). Idiom splits are new local MWE-disjoint splits because official test gold is unavailable. Its separate `unlabeled_official_test.csv` has 916 English texts and must not enter labeled evaluation. Consult `src/dataset_catalog.py` for task meanings; do not treat humor/idiom labels as sarcasm gold or combine them into exclusive classes.

For idiomaticity, supply `mwe` to identify the target expression. `context` holds the preceding sentence and `following_context` the following sentence. Keep `source_label` and human annotations out of model inputs. Unlike the original sarcasm datasets, these new tasks retain sarcasm hashtags because they do not reveal humor/idiom gold labels.

```python
from pathlib import Path
import pandas as pd

folder = Path('Data/Processed/semeval2018')
train = pd.read_csv(folder / 'train.csv', keep_default_na=False)
dev = pd.read_csv(folder / 'dev.csv', keep_default_na=False)
X_train, y_train = train['text'], train['label']
X_dev, y_dev = dev['text'], dev['label']
```

Use `text` as the response-only input and `label` as the target. Keep capitalization, emojis and punctuation for linguistic features. Fit model vectorizers, feature scalers and classifiers only on training data. Tune with development positive-class F1; load test data for final evaluation after decisions are frozen. The schema also retains `id`, `irony_type`, JSON-list `context`, `source`, `platform`, `original_text` and available metadata. Subtypes and rephrasings are label-related metadata and must not enter ordinary target-only classifiers. Context experiments must use the same rows and splits for both model settings.

Use SemEval Task B codes **2 = situational irony; 3 = other verbal irony**, following the official README. Authors/threads/rephrasings/duplicates are grouped where observable; see the methodology for unavailable identifiers and near-duplicate limitations. All official tests are retained, except that SARC explicitly uses a sampled official test by default. Do not compare its sample score to published full-test numbers.

## SARC size experiments

Prepare a larger base pool once, then generate nested training subsets while retaining that base development/test split:

```powershell
.\.venv\Scripts\python.exe src/preprocess.py --dataset sarc --sarc-sample-size 70000 --sarc-test-size 2000
.\.venv\Scripts\python.exe src/sample_training.py --sizes 10000 20000 50000
```

The source pool must exceed the largest desired training size because of the 20% dev reservation and cleaning. The subset command rejects sizes larger than the actual prepared training set. Increasing the pool can require substantially more CPU/RAM for duplicate analysis; the default 10,000 + 2,000 selection is the locally validated configuration. Use `--sarc-test-size 0` only when full official-test preparation is feasible. Changing the source pool regenerates dev; use nested subsets of a fixed base for valid learning curves.

## Transformer baseline

`src/train_bert.py` fine-tunes `cardiffnlp/twitter-roberta-base` (or `roberta-base` via `--model`) on the `text` column only. Torch and transformers are kept out of `requirements.txt` so data preparation stays torch-free:

```bash
pip install -r requirements.txt -r requirements-bert.txt
```

Run the three steps in order, after preparing the dataset:

```bash
python src/train_bert.py --dataset semeval2018 --model cardiffnlp/twitter-roberta-base --search
python src/train_bert.py --dataset semeval2018 --model cardiffnlp/twitter-roberta-base --final --seeds 42 43 44
python src/train_bert.py --dataset semeval2018 --model cardiffnlp/twitter-roberta-base --eval-test
```

`--search` tries learning rates 1e-5/2e-5/3e-5 and batch sizes 16/32 for up to 5 epochs, scored on dev positive-class F1 (ties broken by macro F1), and freezes the choice in `selected_config.json`. It refuses to overwrite an existing frozen config unless `--force` is given. `--final` retrains that config with each seed and keeps the best dev epoch. Only `--eval-test` reads `test.csv`: it requires the frozen config and the final checkpoints, and scores each seed once. A repeat evaluation of the same seed is refused. `--limit N` uses the first N train and dev rows for debugging and writes to a separate `<model>-limitN` folder, which is gitignored.

Runtime for SemEval-2018 with `twitter-roberta-base`: on an Apple M1 Pro (MPS) the search took about 22 minutes, the three seeds about 11 minutes and the test evaluation under a minute. A CUDA GPU such as a Colab T4 should be somewhat faster with mixed precision (roughly 15-25 minutes end to end; not measured). CPU-only training is not practical.

Outputs land in `results/bert/<dataset>/<model_short_name>/`: `search.csv`, `selected_config.json`, `train_log_seed{N}.csv`, `predictions_{dev,test}_seed{N}.csv` (`id,label,pred,prob_positive`), `subtype_recall_dev.csv`, `test_eval_log.csv` and `summary.json`. The best-epoch weights go to `checkpoints/bert/` (gitignored). Tests run on CPU with a tiny random model: `python -m unittest tests.test_train_bert -v`.

**Colab.** With no local GPU, select a GPU runtime and run:

```bash
!git clone https://github.com/hma40/NLP-Group33.git
%cd NLP-Group33
!pip install -r requirements.txt -r requirements-bert.txt
!python src/download_data.py --dataset semeval2018
!python src/preprocess.py --dataset semeval2018
!python src/train_bert.py --dataset semeval2018 --search
!python src/train_bert.py --dataset semeval2018 --final --seeds 42 43 44
!python src/train_bert.py --dataset semeval2018 --eval-test
```

Copy `results/bert/` from the Colab session back into your local checkout before committing. Do not copy `checkpoints/`.

## Outputs and ownership

- [Preparation checklist](DATA_PREPARATION_CHECKLIST.md): current progress and validation status.
- [Dataset statistics](results/dataset_statistics/comparison.csv): report table with platform, annotation, loaded/processed sizes, splits, positive percentage, context, subtypes and limitations.
- `results/dataset_statistics/*.json`: detailed per-dataset missing-value, duplicate, conflict, cleaning and split audits, source-size metadata, and output checksums.
- `results/dataset_statistics/cross_dataset*_overlap.csv`: transfer-experiment overlap diagnostics. FigLang Reddit and the SARC sample share 177 normalized targets; filter source training against target evaluation in a separately documented derived experiment before interpreting transfer scores.
- `results/figures/`: class distributions, length histograms, split sizes and size comparison. SARC charts describe the sample.
- `Notebooks/01_...` through `08_...`: individual exploration and consolidated comparison (06), including humor (07) and idioms (08). Notebooks show aggregates, avoiding restricted text in outputs. The original `exploration.ipynb` is preserved.

Model placeholders owned by Yumi, Sai, Solus and the group are preserved. No weights, corpus text, commits or pushes are created for sharing. Only aggregate statistics, code, plots, source manifests and documentation are intended for repository review. The nikesh66 results are exploratory: its 5,000 input rows yield 224 nonconflicting unique texts and are split by template, with unverified provenance.
