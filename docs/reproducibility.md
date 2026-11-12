# Reproducibility Notes

## Maintenance version 1.0.1

Install the source checkout with all verification extras and run the regression suite.

```bash
python -m pip install -e ".[dev,io,plot]"
python -m pytest -q
python -m build
python -m twine check dist/*
```

The base dependency installation also supports the core tests. External TSPLIB reading uses the `io` extra and plotting uses `plot`. CI is configured for Linux and Windows, builds distributions, and installs the wheel before running all tests and bundled tasks outside the checkout. The matrix covers Python 3.10 through 3.14. A separate job tests the minimum declared NumPy, scikit-learn and matplotlib versions.

### Continuous evaluation budgets

```bash
mmao continuous --problem sphere --dimension 5 --iterations 100 --max-evaluations 500 --seed 7 --output outputs/sphere.json
python examples/benchmark_continuous.py --problem sphere --dimension 5 --max-evaluations 500 --seeds 1 2 3 4 5 --output outputs/batch.json
```

`nfev` counts initialization, sensing, movement, branching and respawn objective calls. The budget must cover the initial population. A movement needs three evaluations and may leave unused budget when fewer calls remain. `termination_reason` identifies early budget stopping or completion of the iteration limit. Other task interfaces currently stop by iterations and retain their domain-specific evaluation behavior.

The batch example reports per-seed results, a mean and sample standard deviation. With multiple seeds it provides a Student's t interval for the mean under an independent, approximately normal sampling assumption. Strongly skewed results require additional analysis. The journal's full paired-testing procedure remains part of the reproduction work in `paper-coverage.md`.

### Result schema and compatibility

Public API and CLI results contain `metadata` with the software version, Python/platform and dependency versions, complete optimizer configuration, seed, elapsed seconds and a problem fingerprint. Classification fingerprints incorporate input data. Continuous fingerprints describe the name, dimensions, bounds and known optimum. Keep the objective function code and its revision separately.

The existing result keys remain available. `optimization_direction` records `minimize` or `maximize`. `gap` is observed minus optimum for minimization, and optimum minus observed for maximization. `relative_gap` divides this value by `max(abs(optimum), 1e-12)`. Knapsack users upgrading from 0.2.0 should account for the corrected sign. A negative gap can indicate a stale supplied reference optimum. Near-zero optima make relative gaps difficult to interpret.

Custom continuous problems default to an unknown optimum. Sphere computes its constrained optimum for the supplied interval. Rastrigin supplies zero when the interval contains zero and otherwise leaves its constrained optimum unknown. Classification reuses the selected validation mask when fitting the final model. On small datasets, KNN neighbor counts are capped at the training subset size, and the final model uses that same cap. Use `return_model=True` to obtain a pipeline accepting the complete input feature matrix, and `save_classification_model` to save it separately. JSON export omits the fitted `model` object.

Result JSON and saved models are replaced atomically. Failed serialization preserves an existing output file. JSON output requires finite numeric values.

Classification history reports `mean_feature_ratio` over surviving agents with evaluated masks. `evaluated_population` gives that count, excluding newly created agents awaiting evaluation. A zero count is represented by a zero mean. `mean_role` covers the entire current population.

### Custom problems and plots

Run `python examples/custom_tasks.py` for a custom objective and array-based classification. `mmao.plotting.plot_history(result, output_path="outputs/history.png")` plots available objective, resource, population and role traces. Install the plotting extra before calling it.

### Release preparation

The repository contains a tag-triggered trusted-publishing workflow. Configure the GitHub `pypi` environment and a PyPI trusted publisher for `.github/workflows/publish.yml` before using it. A manual workflow run builds artifacts, while a version tag also enables publication. Publication depends on successful completion of the reusable CI matrix, the minimum-dependency checks, and the artifact build. The workflow also checks tag/version agreement. See `release-readiness.md` for local validation results and remaining remote checks.

See [paper-coverage.md](paper-coverage.md) for the current scope of correspondence with the published journal experiments.

## Installation

```bash
pip install -e .[dev]
```

Optional loader support:

```bash
pip install -e .[io]
```

## Minimal Verification Commands

Continuous:

```bash
mmao continuous --problem rastrigin --dimension 10 --iterations 80 --seed 7 --summary-only
```

TSP:

```bash
mmao tsp --iterations 100 --seed 11 --summary-only
```

Dynamic:

```bash
mmao dynamic --problem dyn-rastrigin --dimension 10 --iterations 80 --seed 7 --summary-only
```

Classification:

```bash
mmao classification --dataset breast_cancer --classifier svm_rbf --iterations 12 --seed 3 --summary-only
```

## Saving Results

All CLI tasks support:

```bash
mmao continuous --problem sphere --dimension 5 --iterations 20 --output outputs/sphere.json
```

This writes the full JSON result, including the per-iteration history, to the requested path.

## Standard Data Loading

The OR-Library loader supports the `mknap2` collection layout, including its introductory WEING1 example. The bundled file therefore yields 49 entries, beginning with the 28-item example and then the 48 listed instances. This preserves existing numeric indices. Record the instance index, dimensions and known optimum with results.

External TSPLIB files can contain arbitrary tours. Supply a verified reference value explicitly, for example `load_tsplib_problem(path, known_optimum=7542)`. The bundled Berlin52 loader supplies its reference value of 7542.

TSPLIB:

```bash
mmao tsp --tsplib path/to/berlin52.tsp --iterations 150 --summary-only
```

Bundled TSPLIB benchmark:

```bash
mmao tsp --benchmark berlin52 --iterations 150 --summary-only
```

OR-Library multidimensional knapsack:

```bash
mmao knapsack --orlib path/to/mknap2.txt --instance-index 0 --iterations 80 --summary-only
```

Bundled OR-Library benchmark:

```bash
mmao knapsack --benchmark mknap2 --instance-index 0 --iterations 80 --summary-only
```

## Verification Scope

The repository currently verifies:

- editable installation
- command-line execution
- Python API smoke coverage
- source and wheel builds

For paper-scale experiments, this repository should be treated as the unified implementation base and then extended with experiment scripts, batch runners, and post-processing specific to each study.

## Copyright and license

Copyright (c) 2026 Jinliang Xu.

Author and contact — Jinliang Xu, `jlxufly@gmail.com`, `xujinliang@caict.ac.cn`.

This project is licensed under the Apache License, Version 2.0. Full terms and attribution are available in the LICENSE and NOTICE files.
