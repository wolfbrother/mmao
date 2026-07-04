# MMAO Paper Coverage

This repository is the unified software companion for the current MMAO paper family. It is intentionally organized around reusable software modules rather than paper-specific directory mirroring.

## Core Coverage

- Foundational MMAO paper:
  continuous and discrete metabolic control loops are represented by `mmao.continuous` and `mmao.discrete`
- Minimal MMAO paper:
  the package preserves the small closed-loop controller view through lightweight defaults and compact demos
- Large-scale validation paper:
  the repository provides common entry points that can be expanded into stricter benchmark suites
- Dynamic MMAO paper:
  dynamic environment response is represented by `mmao.dynamic`
- Classification MMAO paper:
  mixed discrete-continuous feature selection and classifier tuning are represented by `mmao.classification`
- Mechanism paper:
  the shared code structure makes the metabolic loop explicit across domains

## Why The Repository Is Unified

The papers emphasize different scientific questions, but external users should not need to navigate multiple independent codebases. This repository acts as the common software surface for:

- installation
- quick verification
- standard problem loading
- lightweight reproducibility
- future benchmark expansion

## Recommended Citation Pattern

- Cite this repository or package when referring to software availability, implementation, or reproducibility assets.
- Cite the foundational MMAO paper for the algorithmic origin.
- Cite the relevant derivative paper when discussing a specialized domain such as dynamic optimization or classification-oriented search.
