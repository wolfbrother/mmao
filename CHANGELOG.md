# Changelog

## 1.0.1 — prepared 2026-11-12

- Add Jinliang Xu copyright headers and both contact addresses to Python source files and project configuration, and ship the copyright NOTICE with distributions.
- Align package and citation metadata with the existing Apache License 2.0 text in LICENSE.

- Archive continuous best evaluations, including final-generation children, and preserve TSP best-so-far routes across population turnover.
- Reuse repaired classification masks for final evaluation, expose selected feature indices and classifier settings, and optionally return/export a fitted pipeline.
- Accept array-based custom classification problems.
- Fix population slot accounting in knapsack and dynamic optimization.
- Validate dimensions, bounds, population settings, budgets and core numeric inputs.
- Correct maximization gap direction and default custom continuous optima to unknown.
- Use public APIs in the CLI, add `--version` and continuous `--max-evaluations`, and include evaluation counts and run metadata.
- Resolve bundled Berlin52 without the optional TSPLIB dependency and check file existence before loading external dependencies.
- Add the IEEE Access citation, BibTeX, Chinese quick start, custom task and seed-sweep examples, and optional resource trajectory plots.
- Add defect regression tests, a Linux/Windows CI matrix, wheel-install checks and a trusted-publishing workflow.
- Refresh dynamic personal-best values after environment changes and report initial offline error for zero-iteration runs.
- Validate numeric parameter types, allow zero offspring counts, accept NumPy inputs in metadata, and support small custom KNN datasets.
- Require explicit known optima for external TSPLIB files, use TSPLIB nearest-integer rounding, and validate incomplete OR-Library inputs.
- Preserve output files on serialization failures with atomic JSON/model replacement, and report invalid CLI input through argparse.
- Check Python 3.10–3.14 and minimum dependencies before publication. Use Student's t intervals in the seed-sweep example and repair public documentation links.
- Exclude unevaluated newborn masks from classification feature-ratio summaries and record evaluated population sizes and mean roles.

Results can change after the correctness fixes. Preserve the package version and configuration when comparing runs. Full journal-table reproduction remains a follow-up described in `docs/paper-coverage.md`.

## 0.2.0

- Fixed CLI JSON serialization for classification results containing NumPy arrays.
- Improved README quick-start guidance to distinguish bundled examples, real local file paths, and repository-local scripts.
- Added friendlier loader errors for missing TSPLIB and OR-Library paths.
- Allowed short bundled names such as `berlin52.tsp` and `mknap2.txt` to resolve automatically in the Python API.
- Added regression coverage for CLI classification output and bundled benchmark path handling.

## 0.1.0

- Established the first unified `mmao-opt` package layout.
- Added continuous, TSP, multidimensional knapsack, dynamic, and classification-oriented MMAO interfaces.
- Added CLI entry points, smoke tests, and build validation.
- Added TSPLIB and OR-Library data loaders.
- Added JSON result export utilities and citation metadata.

## Copyright and license

Copyright (c) 2026 Jinliang Xu.

Author and contact — Jinliang Xu, `jlxufly@gmail.com`, `xujinliang@caict.ac.cn`.

This project is licensed under the Apache License, Version 2.0. Full terms and attribution are available in the LICENSE and NOTICE files.
