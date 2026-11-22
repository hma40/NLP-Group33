# Humor and idiom expansion — November 21, 2026

## Decision

Added **HaHackathon (SemEval 2021 Task 7)** for humor and **English Subtask A of SemEval 2022 Task 2** for idiomatic versus non-idiomatic usage. Both have published task papers and human annotations. Retained the original five datasets; SemEval 2018 remains the primary irony benchmark. Keep nikesh66 as a low-priority exploratory dataset because of its repetitive texts and uncertain provenance.

The professor's comment identifies a scope/interpretation issue; it does not explicitly require classifying every form of figurative language. The recommended design retains the main sarcasm/irony question and adds separate auxiliary humor and idiomaticity experiments. Humor is not uniformly nonliteral. Humorous and idiomatic texts can also be sarcastic, and these sources do not jointly annotate all those properties.

**Do not merge these datasets into exclusive sarcasm/humor/idiom/literal classes.** Humor-positive does not mean sarcasm-negative; humor-negative does not mean literal. An idiomaticity label concerns a specified multiword expression (MWE), not all figurative language in the sentence. A combined classifier would need an explicit shared annotation scheme and verified multilabel annotations.

## Sources and suitability

| Addition | Source | Reason for selection | Terms checked |
|---|---|---|---|
| HaHackathon | [Organizer page](https://smash.inf.ed.ac.uk/resources/), [official ZIP](https://smash.inf.ed.ac.uk/hahackathon_data/hahackathon_data.zip), [paper](https://aclanthology.org/2021.semeval-1.9/) | 10,000 short English Twitter/joke texts, annotated by 20 people per text, ages 18–70. Binary humor detection closely complements the existing short-text tasks. | ZIP contains three CSVs and no explicit license file; organizer resource listing supplies no explicit dataset license. Keep texts local; public availability does not establish unrestricted redistribution rights. |
| SemEval 2022 Task 2 English | [Organizer repository](https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity), [paper](https://aclanthology.org/2022.semeval-1.13/) | Human-annotated contextual MWE usage, including idiomatic and non-idiomatic readings. Tests whether models distinguish usage rather than simply recognize a phrase. | Repository has GPL-3.0 LICENSE; checked files do not supply a separate explicit per-text license. Preserve upstream notices, keep third-party texts local and clarify rights before redistribution. No upstream evaluator code was copied. |

The manifest now contains **29 files**: the original 22 plus six selected idiom files and the humor ZIP. Idiom source files are pinned to commit `872625c8f9c5d656b7866ff063a2ab0238ac4c80`, verified from the official GitHub archive's commit comment. Humor has no published revision identifier; its exact archive bytes are SHA-256 pinned. Existing `.gitignore` rules protect all new corpus texts. No additional Python dependencies were needed.

## Measured outputs

| Dataset | Loaded labeled | Prepared | Train | Dev | Test | Positive class |
|---|---:|---:|---:|---:|---:|---|
| HaHackathon | 10,000 | 9,996 | 7,996 | 1,000 | 1,000 | Humorous |
| English idioms | 3,793 | 3,793 | 3,043 | 378 | 372 | Idiomatic usage of the specified MWE |

These are preprocessing statistics, not classifier results. Per-dataset audit JSONs report actual class distributions, missing values, lengths, near duplicates, exclusions and CSV checksums.

### Humor

The original labeled splits contain 8,000 / 1,000 / 1,000 examples. Preserve official development/test roles. Four training rows link to held-out text/near-duplicate components and are excluded; all development and test rows remain. Boundary whitespace is trimmed; `original_text` retains the original target.

`label` directly copies `is_humor`: **1 humorous, 0 non-humorous**. Keep humor rating, controversy and offense rating only as analysis annotations, not predictors. Missing humor ratings/controversy for 3,821 non-humorous examples are conditional missing annotations, not false judgments. The mixed Twitter/joke source is a domain limitation; the release does not identify each row's origin. Sarcasm hashtags and `/s` remain because they are not this task's gold labels.

### Idioms

Load 3,327 English zero-shot training records and 466 English official development records. Join development inputs and gold by **ID and Language**, verifying one-to-one full coverage. Keep source `DataID` and `original_partition`. Exclude trial/one-shot variants to avoid duplicate releases and exposure of held-out expressions.

Paper Table 2 defines **source 0 = idiomatic, 1 = non-idiomatic**, including proper names. Convert to project **1 = idiomatic usage**, and retain `source_label` for validation. `mwe` specifies the expression being classified and is a legitimate model input. Annotation labels and source IDs are not predictors.

The official test contains **916 English records without gold labels**. Preserve them separately in `unlabeled_official_test.csv`; do not invent labels from the submission template or include this file in labeled metrics.

Create new approximately 80/10/10 seed-42 local splits from all 3,793 labeled examples. Keep whole MWEs, shared target/context sentences and near duplicates together. This evaluates unseen-expression generalization and gives group separation priority over exact proportions. The local test is **not official SemEval test gold**; these scores must not be presented as official shared-task results. Combining the former training/development pools also means this is a new local protocol, not the published zero-shot protocol.

`text` is the target sentence. JSON-list `context` contains only Previous; `following_context` separately preserves Next. Both context directions enter leakage grouping. A context model should document using Previous + Target + Next with the MWE. These are web texts, not verified social-media posts; domain differences are an explicit limitation.

## Reproduction and changed files

```powershell
.\.venv\Scripts\python.exe src/download_data.py --dataset hahackathon
.\.venv\Scripts\python.exe src/download_data.py --dataset semeval2022_idioms
.\.venv\Scripts\python.exe src/preprocess.py --dataset hahackathon
.\.venv\Scripts\python.exe src/preprocess.py --dataset semeval2022_idioms
.\.venv\Scripts\python.exe src/explore_data.py
.\.venv\Scripts\python.exe src/validate_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--dataset all` now includes both additions. New `src/figurative_datasets.py` implements the loaders; `src/dataset_catalog.py` declares all task/positive/negative meanings. Existing downloader, preprocessing, reporting and validation commands support them. New notebooks 07 (humor) and 08 (idioms) accompany the refreshed comparison notebook 06. The comparison CSV now explicitly identifies each task and its label meanings.

## Next experiments

1. Finish the original SemEval 2018 baseline/feature comparisons first.
2. Run the same model families separately on humor and idiomaticity, tuning only on their development splits. Supply the MWE for the idiom task; compare sentence-only and context models on identical rows.
3. Investigate whether sentiment/punctuation features help differently across tasks. F1 differences alone do not isolate category effects because domains, sizes and annotation methods differ.
4. Use development examples for cross-category error analysis. To measure sarcasm false positives on humor/idiom examples, first obtain independent sarcasm annotations; existing labels are insufficient.
5. Reserve each held-out test for frozen final experiments. A combined figurative-language task remains future work requiring joint annotation.

No classifiers were trained, old dataset files replaced, commits created or changes pushed.

## Validation completed

- All 29 source checksums pass. All eight prepared portions (24 main split CSVs) pass schema, label, ID, text/group/metadata separation, count and output-hash checks.
- Existing split files continue to match their previously recorded hashes; original model placeholders remain unchanged.
- Official humor test IDs, original targets and labels match the organizer archive. The 193 idiom MWEs occupy disjoint train/dev/test sets of 156 / 19 / 18 expressions.
- Eight regression tests pass, including new checks for idiom label inversion, context direction, retained humor hashtags and official development membership.
- Both new notebooks and refreshed comparison notebook executed successfully; new figures were visually inspected. Expanded cross-dataset normalized/exact-target audits find no overlaps involving either new corpus. This does not establish an absence of semantic or source-document overlap.

## References

- J. A. Meaney, Steven Wilson, Luis Chiruzzo, Adam Lopez, and Walid Magdy. 2021. [SemEval 2021 Task 7: HaHackathon, Detecting and Rating Humor and Offense](https://aclanthology.org/2021.semeval-1.9/). In *Proceedings of the 15th International Workshop on Semantic Evaluation*, pages 105–119, Online. Association for Computational Linguistics.
- Harish Tayyar Madabushi, Edward Gow-Smith, Marcos Garcia, Carolina Scarton, Marco Idiart, and Aline Villavicencio. 2022. [SemEval-2022 Task 2: Multilingual Idiomaticity Detection and Sentence Embedding](https://aclanthology.org/2022.semeval-1.13/). In *Proceedings of the 16th International Workshop on Semantic Evaluation*, pages 107–121, Seattle, United States. Association for Computational Linguistics.
