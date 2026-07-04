# mmao

MMAO is a metabolic multi-agent optimization framework for continuous, discrete, dynamic, and classification-oriented search. It uses endogenous resource allocation and closed-loop adaptation to provide a reusable, parameter-light optimizer with reproducible benchmarks, examples, and research-friendly APIs.

## What This Repository Provides

This repository is the unified software release for the MMAO research line. It consolidates the core ideas validated across the current MMAO papers into one installable Python package, with a shared API and lightweight command-line entry points.

The current package includes:

- Continuous optimization with a reusable MMAO controller
- Discrete optimization for TSP and multidimensional knapsack
- Dynamic optimization with environment-shift response
- Classification-oriented optimization for feature selection and classifier tuning
- Standard loaders for TSPLIB and OR-Library style benchmark data
- JSON result export, smoke tests, and a PyPI-ready package layout

## Why This Repository Matters

The MMAO papers address different scientific questions, but external readers should not have to navigate several disconnected codebases. This repository is the common implementation surface that the papers can cite for:

- installation and quick verification
- unified interfaces across domains
- lightweight reproducibility
- standard data loading
- future benchmark expansion

## Design Goal

This repository is intended to make MMAO easier to verify, test, and reuse. It is not just a dump of paper scripts. The package is organized around a common metabolic control logic:

- agents maintain private energy
- a communal resource pool redistributes search pressure
- role states shift continuously with success and scarcity
- expansion, contraction, and respawn are driven by the same internal loop

## Installation

Install from PyPI with `pip`:

```bash
pip install mmao-opt
```

Install from PyPI with `uv`:

```bash
uv pip install mmao-opt
```

If you use `uv` as your project manager, you can also add the package as a dependency:

```bash
uv add mmao-opt
```

For local development with `pip`:

```bash
pip install -e .
```

For local development with `uv`:

```bash
uv pip install -e .
```

If you want TSPLIB loading support with `pip`:

```bash
pip install -e .[io]
```

If you want TSPLIB loading support with `uv`:

```bash
uv pip install -e ".[io]"
```

For development and testing with `pip`:

```bash
pip install -e .[dev]
```

For development and testing with `uv`:

```bash
uv pip install -e ".[dev]"
```

## Quick Start

Run the packaged CLI:

```bash
mmao continuous --problem rastrigin --dimension 10 --iterations 80
mmao tsp --iterations 100
mmao dynamic --problem dyn-rastrigin --dimension 10 --iterations 80
mmao classification --dataset breast_cancer --classifier svm_rbf --iterations 12
```

To print a compact summary instead of the full history:

```bash
mmao continuous --problem sphere --dimension 5 --iterations 20 --summary-only
```

To save the full JSON result:

```bash
mmao continuous --problem sphere --dimension 5 --iterations 20 --output outputs/sphere.json
```

Standard data loaders are also available:

```bash
mmao tsp --tsplib path/to/berlin52.tsp --summary-only
mmao knapsack --orlib path/to/mknap2.txt --instance-index 0 --summary-only
```

The package also bundles two standard benchmark files for direct verification after installation:

```bash
mmao tsp --benchmark berlin52 --summary-only
mmao knapsack --benchmark mknap2 --instance-index 0 --summary-only
```

You can also run the Python example:

```bash
python examples/quickstart.py
```

## Python API

```python
from mmao.api import optimize_continuous
from mmao.continuous import MMAOContinuousConfig, rastrigin_problem

problem = rastrigin_problem(dimension=10)
result = optimize_continuous(problem, MMAOContinuousConfig(iterations=80, seed=7))
print(result["best_fitness"])
```

For file-based benchmark loading:

```python
from mmao import load_tsplib_problem, load_orlib_mkp_instance, optimize_tsp, optimize_knapsack

tsp_problem = load_tsplib_problem("berlin52.tsp")
mkp_problem = load_orlib_mkp_instance("mknap2.txt", index=0)
```

## Package Structure

```text
mmao/
  continuous/      continuous optimization
  discrete/        tsp and knapsack
  dynamic/         dynamic optimization
  classification/  feature selection and classifier tuning
  io/              optional loaders such as TSPLIB
  cli.py           command-line interface
```

## Scope of This First Release

This first release focuses on a clean and usable public package interface. It already exposes the major MMAO problem families, but it is still a research-oriented release rather than a fully stabilized production library.

Near-term improvements will likely include:

- richer benchmark adapters
- experiment orchestration utilities
- result serialization helpers
- stronger dataset and problem loaders
- broader documentation for paper-level reproducibility

## Testing

Run the smoke tests with:

```bash
pytest
```

The repository currently includes:

- API smoke tests
- CLI smoke tests
- data-loader tests
- source and wheel build validation

For public trust and regression control, CI is defined in `.github/workflows/ci.yml`.

## Citation

Please use [CITATION.cff](D:/WorkFiles/VscodeProject/ED-RSA/mmao/CITATION.cff) for software citation metadata.

Recommended practice:

- cite the repository or `mmao-opt` package for software availability and reproducibility
- cite the foundational MMAO paper for the core optimizer
- cite the relevant derivative paper for domain-specific discussion such as dynamic optimization or classification-oriented search

Additional orientation documents:

- [docs/paper-coverage.md](D:/WorkFiles/VscodeProject/ED-RSA/mmao/docs/paper-coverage.md)
- [docs/reproducibility.md](D:/WorkFiles/VscodeProject/ED-RSA/mmao/docs/reproducibility.md)
