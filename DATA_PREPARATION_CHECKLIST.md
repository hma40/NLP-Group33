# Dataset preparation progress

All work is local; do not commit or push raw/processed social media text.

- [x] Inspect repository; preserve empty model placeholders and original exploration notebook.
- [x] Check Python 3.12 and existing `.venv`; no BERT dependencies installed.
- [x] Verify sources, current schemas, official test labels and source restrictions.
- [x] Complete SemEval 2018 first: verified Task A/B join, splits, audit and notebook.
- [x] Complete English iSarcasmEval with annotations and rephrasing protection.
- [x] Complete FigLang Twitter and Reddit separately, preserving labeled official tests.
- [x] Complete official SARC 2.0 balanced sample with streaming context lookup.
- [x] Complete nikesh66 with conflict removal and template-group splitting.
- [x] Validate all CSVs, leakage checks, statistics, plots and reproducibility.
- [x] Document teammate commands, dataset limitations and final observed counts.

See `docs/DATASETS.md` for provenance and decisions, and generated JSON audits for actual removals. Sampled SARC statistics must never be described as full-corpus statistics.

## Validated outcomes

- All 22 raw-source checksums pass; pinned download command and offline verification supplied.
- All 18 main CSVs reload and pass schema, binary label, nonempty input, ID, group, text-overlap, metadata-link, count and checksum checks.
- Full official SemEval/FigLang test membership, original targets and labels checked against raw sources; SARC sampled test identifiers/labels checked against official sequence records.
- All six exploration notebooks executed successfully in the project environment. Seven aggregate chart PNGs generated; primary dataset and comparison figures visually inspected for layout.
- Five regression tests cover minimal cleaning, transitive context/rephrasing/author links, duplicate bridge preservation, seeded stratification and bounded seeded reservoir sampling.
- Nested 1,000/5,000-example SARC training subsets created; dev/test files remain fixed.
- Cross-dataset overlap diagnostics saved; FigLang Reddit/SARC share 177 normalized targets. Future transfer experiments require derived source-training exclusions.
- No classifiers, model weights, commits or pushes created. Full-corpus SARC and larger source samples have not been run; the bounded default is complete.

## November 21: humor/idiom expansion

- [x] Research academic sources and inspect official releases/terms.
- [x] Add HaHackathon humor, preserve official dev/test, exclude four linked training rows.
- [x] Add English SemEval 2022 Task 2; verify reversed source label mapping and ID-joined dev gold.
- [x] Prepare MWE-disjoint local idiom splits and retain 916 official test records as explicitly unlabeled.
- [x] Extend download manifest (29 files), task catalog, preprocessing, exploration and validation.
- [x] Add exploration notebooks 07/08; refresh consolidated comparison.
- [x] Expand regression tests from five to eight, all passing.
- [x] Document separate task meanings and the limits of cross-category evaluation.

See docs/FIGURATIVE_EXPANSION.md. Original datasets and model placeholders retained; no joint-label dataset or classifier results fabricated.
