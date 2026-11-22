# Observed preparation results

Generated from audit files. SARC rows describe the default selected sample, not the whole corpus.

Tasks stay separate: positive means irony/sarcasm, humor, or idiomatic usage according to the dataset. Idiom test is a local MWE-disjoint holdout, not official gold test.

| Dataset | Loaded | Processed | Train | Dev | Test | Positive % |
|---|---:|---:|---:|---:|---:|---:|
| semeval2018 | 4618 | 4618 | 3067 | 767 | 784 | 48.12 |
| semeval2022 | 4868 | 4850 | 2761 | 689 | 1400 | 21.90 |
| figlang2020/twitter | 6800 | 6546 | 3797 | 949 | 1800 | 51.89 |
| figlang2020/reddit | 6200 | 6197 | 3515 | 882 | 1800 | 50.01 |
| sarc | 12000 | 11749 | 7802 | 1947 | 2000 | 49.80 |
| nikesh66 | 5000 | 224 | 162 | 31 | 31 | 50.00 |
| hahackathon | 10000 | 9996 | 7996 | 1000 | 1000 | 61.79 |
| semeval2022_idioms | 3793 | 3793 | 3043 | 378 | 372 | 51.25 |

Use comparison.csv for annotation methods and limitations; per-dataset JSON for missing values, conflicts, lengths and removals.
Cross-dataset overlap tables are diagnostics. Before transfer experiments, exclude target-evaluation overlaps on the source-training side using a separate derived split; retain the original benchmarks.
