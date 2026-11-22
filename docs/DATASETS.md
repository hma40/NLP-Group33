# Dataset provenance and preparation decisions

Original sources checked on November 20, 2026; humor/idiom sources added November 21. See [the expansion report](FIGURATIVE_EXPANSION.md) for new sources, terms, statistics and experimental design. `config/dataset_sources.json` pins GitHub/Hugging Face revisions and SHA-256 checksums for 29 downloaded files (22 original plus seven additions). The official SARC archive has no Git revision, so its exact downloaded bytes are pinned by checksum. Downloaded READMEs and license files remain next to raw data for local reference.

## Sources, formats and terms

| Dataset | Selected source and format | Terms and redistribution findings |
|---|---|---|
| SemEval 2018 | [Organizer repository](https://github.com/Cyvhee/SemEval2018-Task3); Task A/B original-emoji, collection-hashtag-excluded TSV training and gold test files | Root README identifies CC BY-NC-SA 4.0 and academic use, with an organizer-contact disclaimer. Training README separately prohibits redistribution of Twitter data/annotations under platform terms. Keep corpus files local; cite organizers and consult them before any redistribution. |
| iSarcasmEval English | [Organizer repository](https://github.com/iabufarha/iSarcasmEval); `train/train.En.csv`, `test/task_A_En_test.csv` | Repository includes MIT LICENSE. Preserve attribution; do not assume a repository software license overrides third-party text/platform rights. This project does not redistribute texts. |
| FigLang 2020 | [Organizer repository](https://github.com/EducationalTestingService/sarcasm); separate Twitter/Reddit training/testing JSONL | No explicit dataset license found in repository root README or file tree. Public download is not evidence of unrestricted redistribution. Keep texts local; obtain clarification for redistribution. |
| SARC | [Official relocated archive](https://nlp.cs.princeton.edu/old/SARC/2.0/); `main/train-balanced.csv.bz2`, `test-balanced.csv.bz2`, `comments.json.bz2` | Archive readme gives format and citation but no explicit dataset license. The [evaluation code repository](https://github.com/NLPrinceton/SARC) has an MIT software license; this does not establish corpus redistribution rights. Keep texts local and clarify corpus terms before sharing. |
| nikesh66 | [Hugging Face dataset](https://huggingface.co/datasets/nikesh66/Sarcasm-dataset); `sarcasm_tweets.csv` | No license, collection procedure or annotation methodology specified in the checked card. No claim of unrestricted redistribution. Exploratory local analysis only. |

The originally linked [CreativeLang SARC mirror](https://huggingface.co/datasets/CreativeLang/SARC_Sarcasm) advertises CC BY 2.0, but its schema contains 12,704,751 comment records and **no label field or official sequence partitions**. Do not infer labels from record order or attach every comment a positive label. We use the official labeled archive instead. The old `/SARC/2.0/` URL returned 404; `/old/SARC/2.0/` works. The [FigLang Hugging Face mirror](https://huggingface.co/datasets/tasksource/figlang2020-sarcasm) has a schema-casting error; loading the official platform-specific JSONL files avoids it.

## Common model-ready contract

Each dataset directory contains `train.csv`, `dev.csv`, `test.csv`, encoded as UTF-8. Required columns are `id`, `text`, `label`, `irony_type`, `context`, `source`, `platform`. Additional fields retain `original_text`, grouping identifiers and available annotations. IDs are unique **within a prepared dataset**, with partition prefixes for source row indices. A FigLang Twitter ID may equal a FigLang Reddit ID; treat those as separate datasets.

`label=1` means irony for SemEval 2018, sarcasm for the other original datasets, humor for HaHackathon, and idiomatic MWE usage for SemEval 2022 idioms. The new CSVs explicitly record task and label meanings; the dataset catalog/comparison table declares all tasks. These are overlapping properties, not exclusive classes. These definitions and annotation procedures differ. They are not interchangeable gold standards. Unknown optional values are empty, not negative observations. Metadata are for auditing and analysis, not automatic model inputs. In particular, irony subtypes, sarcastic rephrasings, author, and collection metadata can reveal labels or change the experimental task.

`context` is a JSON array of preceding messages, oldest to newest; `[]` means unavailable. Decode it with `json.loads(row.context)`. For response-only experiments use only `text`; for context experiments concatenate those messages followed by `text`, with a documented separator. Compare both settings on identical examples/splits.

For the original irony/sarcasm datasets, cleaning strips leading/trailing whitespace and explicit label-revealing `#irony`, `#sarcasm`, `#sarcastic`, `#not`, and standalone `/s` markers, case-insensitively. It applies to targets, context, parents and rephrasings. The marker policy is deliberately narrow; words such as ordinary “not” remain. Original target text stays in `original_text`; full original records remain in `Data/Raw`. Capitalization, punctuation, emojis, repeated characters, stop words and internal whitespace are retained. `markers_removed_rows` counts targets containing a recognized marker; `trimmed_or_marker_cleaned_rows` includes whitespace trimming too.

The added humor/idiom tasks trim boundary whitespace only and retain sarcasm hashtags. For idioms, following context is retained separately from preceding context, and the specified MWE is a legitimate model input. Human labels, ratings and source IDs remain metadata. The expansion report documents local idiom splitting and official humor dev/test preservation.

## Leakage and split policy

1. Validate binary labels, required inputs and source IDs. Join SemEval annotations one-to-one by verified index within the same official partition.
2. Build connected components before deleting rows, so discarded duplicates cannot hide conversation links. Link normalized targets, rephrasings, shared context messages with at least 20 normalized characters, available conversation IDs and nondeleted authors. Also link nikesh66 subject-masked templates.
3. For comparison only, text keys ignore case/punctuation, mask URLs, mentions and digit runs. Emoji-only strings retain distinct comparison keys. These conservative keys can group texts with different meanings; they never replace model input text.
4. Investigate additional near duplicates using character 3–5 gram TF-IDF cosine similarity ≥0.95 on unique target texts. Use these links only for grouping. This is a documented surface-similarity heuristic, not an exhaustive guarantee against semantic duplicates. Full-text TF-IDF here supports split hygiene, not a model feature representation; model vectorizers still must be fit on training data only.
5. Remove empty training targets, training texts with conflicting normalized labels and repeated training target texts. Retain official test rows even if duplicated or conflicting internally. All original records remain locally for audits and future controlled quality experiments.
6. Exclude entire training components linked to official test records. Never move test examples into training, and never silently delete official test examples. With SARC, “official test” here refers to the explicitly selected sample, not the whole release.
7. Split the remaining official training pool 80/20 with seed 42. Use stratification for independent singleton records; otherwise choose among 256 group splits by size and label balance, requiring both classes in each split. Without an official test, split 80/10/10 approximately, prioritizing components over exact proportions. The same seed and source bytes reproduce the same CSVs.
8. Independently reload CSVs and check unique IDs, binary labels, nonempty text, JSON contexts, disjoint groups and normalized targets, reported counts and SHA-256 hashes. Audit JSON records actual counts, class percentages, removals and split percentages.

Available thread IDs and authors are protected where present. FigLang has no original thread or author IDs, and SemEval releases do not supply authors. Shared-context grouping is a proxy and cannot reconstruct every original conversation. Common short acknowledgments and deleted-message placeholders are not used as conversation links. Remaining limitations belong in the final report.

The aggregate cross-dataset audit also finds **177 shared normalized target strings between FigLang Reddit and this SARC sample**, plus one between SARC and each SemEval dataset. Normalized matches may include generic phrases; consult `cross_dataset_overlap.csv` for exact-string counts and `cross_dataset_split_overlap.csv` for partition-level counts. These separate benchmark datasets are intentionally retained. Before transfer or combined-dataset experiments, create derived source-training splits excluding target-evaluation overlaps and document the exclusions. Do not report naive transfer scores as independent generalization.

## Dataset-specific observations

### SemEval 2018: primary benchmark

The checked release contains 3,834 training and 784 gold test tweets. The final split is 3,067 / 767 / 784, retaining every source example. Use Task A as the classification target. Task B codes from the [official training README](https://github.com/Cyvhee/SemEval2018-Task3/blob/master/datasets/train/README.md) are **0 non-ironic, 1 verbal irony with polarity contrast, 2 situational irony, 3 other verbal irony**. This corrects the reversed 2/3 descriptions in the initial project notes. Do not equate these categories exactly with explicit/implicit incongruity.

Index values restart between train and test. Index 2305 in training has a documented source-text difference: Task A uses `|` separators, Task B literal `\n` separators. The loader verifies equivalent normalized wording for precisely this exception, retains Task A text, verifies all other texts exactly, and checks binary/subtype label agreement. It does not assume row order. These indices are released sample indices, not verified Twitter status IDs.

### iSarcasmEval: author-intended sarcasm

The English release contains 3,468 training rows and 1,400 labeled Task A test rows. Retain six expert category columns and author rephrasings when supplied. The Task A test file contains only text and binary labels, so absent test categories/rephrasings remain unavailable. Rephrasings are not expanded into extra negative training rows. Their normalized text still links related rows for splitting, protecting a later paired-text experiment. An absent category annotation is not evidence that the category is false.

One training target is missing; duplicates and contradictory labels also occur. Exact removal counts are in `semeval2022.json`. Author intention differs from third-party judgments of irony; discuss that difference in cross-dataset evaluation.

### FigLang: keep platforms separate

The release contains 5,000 Twitter and 4,400 Reddit training examples, plus 1,800 test examples per platform. Contrary to the historical README's prospective test description, both current upstream testing files include `SARCASM` / `NOT_SARCASM` gold labels, balanced 900/900. Preserve both labeled tests. Their within-test duplicates are retained as part of the official benchmark. Raw files include only response/context and a test sample ID; no verified conversation IDs are invented. Shared context creates group links.

### SARC: scalable source reading, bounded default sample

The selected **main balanced 2.0** sequence files contain 257,082 training responses and 64,666 test responses, exactly balanced within each official partition. These counts are measured by scanning sequence files and are not the size of the entire SARC corpus or the CreativeLang comment table.

The default selects 10,000 training-pool responses and 2,000 official-test responses by separate seed-42 reservoir samples. This is uniform response sampling, not imposed stratification; report observed class balance. A streaming `ijson` reader scans the 447,876,208-byte bzip2 comment archive, retaining only selected targets and ancestors. It neither decompresses the full corpus to disk nor loads the full comment dictionary. The sampled test is not directly comparable with full official-test research scores. Author, subreddit, actual Reddit comment ID, ancestor IDs, root conversation ID and chronological context are retained.

`--sarc-sample-size 20000` or `50000` requests a larger source training pool. Actual training size is smaller after cleaning and reserving development data. `--sarc-test-size` defaults to 2000 independently of training pool size. Changing the pool and rerunning preparation can change development membership: for a learning curve, prepare **one sufficiently large base split once**, then use `sample_training.py` for nested subsets of `train.csv`. This keeps dev/test CSVs fixed. Nested subsets use deterministic SHA-256 priorities and report their class counts. They do not promise exact class balance.

`--sarc-sample-size 0` / `--sarc-test-size 0` disable the corresponding source sample cap. Selected records and near-duplicate comparisons then need substantially more RAM/time; only the comment lookup remains streaming. The bounded default was validated locally. Full-corpus preparation was not run, and the current pairwise surface-similarity pass is not suitable for millions of examples without a scalable candidate index.

### nikesh66: exploratory quality analysis

The actual CSV has 5,000 rows and columns `Tweet`, `Sarcasm (yes/no)`, although its card describes a slang classification column. There are 4,760 repeated-text rows (240 distinct texts). Sixteen normalized texts have conflicting labels, affecting 615 rows. Excluding those conflicting texts and deduplicating leaves **224** usable texts. A large reduction is an observed property, not an estimate.

Subject phrases recur in sentence frames. Grouping masks the observed subject choices (e.g. professions, event names and object phrases) and holds complete resulting templates within splits. The final 162 / 31 / 31 split deviates from 80/10/10 because independent templates and both classes must remain separated. Template-like wording is confirmed; synthetic generation is **not** established. Collection and annotation provenance remain unverified. Use this as an exploratory stress test, not an equally reliable academic benchmark. The raw rows support future before/after quality analyses, but these must still honor common held-out template groups.

## Verified dataset references

- Cynthia Van Hee, Els Lefever, and Véronique Hoste. 2018. [SemEval-2018 Task 3: Irony Detection in English Tweets](https://aclanthology.org/S18-1005/). In *Proceedings of the 12th International Workshop on Semantic Evaluation*, pages 39–50, New Orleans, Louisiana. Association for Computational Linguistics.
- Ibrahim Abu Farha, Silviu Vlad Oprea, Steven Wilson, and Walid Magdy. 2022. [SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic](https://aclanthology.org/2022.semeval-1.111/). In *Proceedings of the 16th International Workshop on Semantic Evaluation*, pages 802–814, Seattle, United States. Association for Computational Linguistics.
- Debanjan Ghosh, Avijit Vajpayee, and Smaranda Muresan. 2020. [A Report on the 2020 Sarcasm Detection Shared Task](https://aclanthology.org/2020.figlang-1.1/). In *Proceedings of the Second Workshop on Figurative Language Processing*, pages 1–11, Online. Association for Computational Linguistics.
- Mikhail Khodak, Nikunj Saunshi, and Kiran Vodrahalli. 2018. [A Large Self-Annotated Corpus for Sarcasm](https://aclanthology.org/L18-1102/). In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan. European Language Resources Association.

The nikesh66 uploader supplies no verified academic dataset citation. Cite the pinned dataset page and access date rather than inventing a paper.
