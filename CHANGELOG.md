# Changelog

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
