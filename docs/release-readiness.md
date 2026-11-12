# Version 1.0.1 release review

Reviewed on 2026-11-12. This maintenance release adds correctness fixes and reusable research interfaces. Release validation covers the package APIs, CLI, serialization, benchmark loaders, metadata and distribution files.

## Correctness and compatibility

- Continuous search archives every evaluated best position, including final-generation offspring. Objective calls are counted and remain within an optional evaluation limit.
- TSP search preserves the best recorded route across agent removal. External TSPLIB tours are treated as tours, with verified optimum values supplied explicitly. Bundled EUC_2D distances use TSPLIB rounding.
- Knapsack and dynamic controllers account for all surviving agents when allocating offspring slots.
- Classification uses the same selected mask for validation and final training. Small-data KNN settings retain the same effective neighbor limit. The fitted pipeline accepts complete feature matrices.
- Dynamic reevaluation refreshes stored personal-best values in the current environment. A zero-iteration run reports the initial offline error.
- Numeric inputs, dimensions, population limits, labels and malformed benchmark files receive explicit validation. Zero offspring counts remain supported.
- JSON and model writes use temporary files followed by atomic replacement. Serialization failure preserves previous output.
- NumPy inputs and scalar seeds work with run metadata and JSON export. Unknown custom optima remain unspecified. Maximization gaps use the documented corrected direction.
- Classification trajectory summaries exclude unassessed offspring masks and record the evaluated population size.

## Local verification

The validation environments are independent virtual environments on Windows, without inherited system packages.

| Environment | Configuration | Result |
| --- | --- | --- |
| Python 3.10.22 | NumPy 1.24.0, scikit-learn 1.3.0, SciPy 1.10.1, joblib 1.2.0, matplotlib 3.7.0 | 82 tests pass |
| Python 3.11.18 | Full extras and current compatible dependencies | 82 tests pass |
| Python 3.12.14 | Base dependencies and test tools | 80 tests pass, 2 optional-dependency tests skip |
| Python 3.12.14 | Full extras | 82 tests pass |
| Python 3.13.18 | Full extras | 82 tests pass |
| Python 3.14.7 | Full extras | 82 tests pass |

The minimum dependency configuration emits an upstream pyparsing deprecation warning and a joblib warning about Windows physical-core detection falling back to logical cores. These warnings accompany successful tests. Dependency compatibility checks pass. Ruff checks Python errors and unused names, and actionlint validates the GitHub Actions workflows.

The source distribution and wheel pass Twine metadata checks. The wheel is installed and tested outside the source directory. The source archive is separately installed and tested. Citation metadata passes CFF 1.2.0 schema validation. The source archive includes the Chinese README, citations, examples and documentation. Both distributions include the bundled benchmark data.

## Release workflow

Tag publication depends on the reusable CI workflow. That workflow tests Python 3.10 through 3.14 on Linux and Windows, verifies base and minimum dependencies, and installs the built wheel before running tests outside the source tree. The publishing workflow also checks agreement between the version tag and installed package version.

Remote Linux jobs and GitHub-hosted jobs must execute successfully on the pushed revision. Linux validation has not run on this local host because its Docker Linux engine is unavailable. Configure the repository's `pypi` environment and PyPI trusted publisher before tagging `v1.0.1`. Local validation does not exercise those account settings or upload an artifact.

## Research scope

The release provides verified software interfaces, examples, descriptive seed-sweep statistics and continuous evaluation budgets. Journal-table reproduction and all-domain evaluation budgets are tracked in `paper-coverage.md`. Users comparing research results should retain the package version, configuration, dataset identity and objective-function revision.

## Copyright and license

Copyright (c) 2026 Jinliang Xu.

Author and contact — Jinliang Xu, `jlxufly@gmail.com`, `xujinliang@caict.ac.cn`.

This project is licensed under the Apache License, Version 2.0. Full terms and attribution are available in the LICENSE and NOTICE files.
