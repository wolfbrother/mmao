# mmao

MMAO is a metabolic multi-agent metaheuristic optimization framework for continuous optimization, combinatorial optimization, dynamic optimization, and classification-oriented search. It uses endogenous resource allocation, closed-loop adaptation, and parameter-light control to provide a reusable Python package for optimization research, algorithm development, and reproducible benchmarking. The repository includes benchmark loaders, command-line tools, example workflows, and research-friendly APIs for tasks such as TSP, multidimensional knapsack, feature selection, and classifier tuning.

The table below summarizes how MMAO differs, at the framework level, from several well-known heuristic and plant-inspired optimizers. The goal is not to rank these methods universally, but to clarify the design dimensions on which MMAO contributes a distinct closed-loop control logic.

| Algorithm | Core Inspiration / Abstraction | Resource Allocation Logic | Population / Agent Lifecycle | Search Guidance | Adaptation Style | Cross-Domain Unification |
|---|---|---|---|---|---|---|
| **MMAO / Metabolic Multi-Agent Optimizer** | Metabolic closed-loop economics of survival and reinvestment | **Endogenous private-public resource loop** with earned, recycled, and reinvested budget | **Dynamic branching, pruning, and respawn** emerge from energy state | Energy-regulated sensing; symmetric probing in continuous spaces and structural sensing in discrete spaces | Unified endogenous adaptation of scale, role, effort, and redistribution | **Yes; same control logic across continuous and discrete domains** |
| **PSO / Particle Swarm Optimization** | Swarm motion and social learning | No endogenous resource economy | Usually fixed population | Velocity update using personal and global best | Usually external or heuristic parameter adaptation | Mainly continuous; discrete variants require redesign |
| **DE / Differential Evolution** | Differential variation in evolving populations | No endogenous resource accounting | Usually fixed population | Differential mutation, crossover, and greedy selection | Usually external parameter adaptation | Mainly continuous; discrete transfer is non-native |
| **APOA / Artificial Plant Optimization Algorithm** | Artificial plant growth and branching | No unified budget loop | Rule-based growth and branching | Plant-like branching rules | Mostly rule-based | Typically domain-specific |
| **TSA / Tree-Seed Algorithm** | Tree growth and seed dispersal | No endogenous reinvestment economy | Static individuals with seed generation | Seed dispersal around parent solutions | Mostly fixed probabilistic adaptation | Mainly continuous |
| **IWO / Invasive Weed Optimization** | Weed colonization and reproduction | No closed resource loop | Rule-driven expansion and reduction | Random dispersal with variance schedule | Schedule-based adaptation | Mainly continuous |
| **RMO / Root Mass Optimization** | Root morphology and branching geometry | No private-public budget closure | Geometry- or probability-driven branching | Root-like extension in search space | Morphology-driven adaptation | Mostly continuous or domain-specific |
| **RGA / Root Growth Algorithm** | Root growth and branching behavior | No endogenous budget economy | Branching governed by root-style rules | Root-direction and branching heuristics | Rule-based adaptation | Mostly continuous or domain-specific |

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
mmao tsp --tsplib /absolute/or/real/local/path/to/berlin52.tsp --summary-only
mmao knapsack --orlib /absolute/or/real/local/path/to/mknap2.txt --instance-index 0 --summary-only
```

The `--tsplib` and `--orlib` arguments expect real local file paths. If you just want a quick verification run after installation, use the bundled benchmarks below.

The package also bundles two standard benchmark files for direct verification after installation:

```bash
mmao tsp --benchmark berlin52 --summary-only
mmao knapsack --benchmark mknap2 --instance-index 0 --summary-only
```

If you cloned the GitHub repository, you can also run the local example script from the repository root:

```bash
cd mmao
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

The short names `"berlin52.tsp"` and `"mknap2.txt"` resolve to the bundled examples shipped with the package. If you want to load your own files, pass a real local file path instead.

## How to Extend MMAO

There are three practical ways to build on `mmao-opt`, depending on whether you want to reuse it as-is, adapt it to new problems, or modify the optimizer itself.

If you only need MMAO as a research tool, you can install the package and call the public API directly. The exported interfaces in [src/mmao/api.py](D:/WorkFiles/VscodeProject/ED-RSA/mmao/src/mmao/api.py) allow you to define your own problem instances and run MMAO without changing the package source. In practice, this is the easiest path for plugging MMAO into custom benchmark scripts, engineering workflows, or evaluation pipelines.

If you want to adapt or improve the implementation, clone the repository and use an editable install:

```bash
git clone https://github.com/wolfbrother/mmao.git
cd mmao
pip install -e .[dev]
```

Or with `uv`:

```bash
uv pip install -e ".[dev]"
```

The current package structure is intended to make domain-specific extensions straightforward:

- `src/mmao/continuous/core.py` for continuous optimization
- `src/mmao/discrete/tsp.py` for TSP
- `src/mmao/discrete/knapsack.py` for multidimensional knapsack
- `src/mmao/dynamic/core.py` for dynamic optimization
- `src/mmao/classification/core.py` for classification-oriented optimization
- `src/mmao/io/` for benchmark and dataset loaders

Typical extension directions include:

- replacing or strengthening sensing and move operators
- refining branching, pruning, respawn, or reinvestment logic
- adding new benchmark adapters and data loaders
- introducing new problem families such as constrained, multi-objective, or scheduling optimization
- reducing manual hyperparameters by deriving more behaviors from the internal metabolic loop

If you plan to publish a derived variant, the most important design recommendation is to preserve the core MMAO identity: private energy, a communal resource pool, and closed-loop adaptation should remain the primary control mechanism. This makes it easier for others to understand the new method as part of the MMAO family rather than as a disconnected heuristic.

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
