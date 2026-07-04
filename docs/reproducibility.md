# Reproducibility Notes

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
