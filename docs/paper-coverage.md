# MMAO Paper Coverage

## IEEE Access publication

Jinliang Xu, Bingqi Li, Liping Ma, Runkai Zhu, Fanjie Nie, and Jin Li, *MMAO: A Unified Resource-Regulated Controller for Cross-Domain Optimization*, IEEE Access, 2026, pp. 1–12. DOI [10.1109/ACCESS.2026.3752083](https://doi.org/10.1109/ACCESS.2026.3752083). Bibliographic fields follow the author-provided IEEE export.

| Task interface | Package implementation | Verification available in this repository |
| --- | --- | --- |
| Static continuous | `src/mmao/continuous/core.py` | Custom objectives, seed sweeps, evaluation counts and budgets |
| Static routing | `src/mmao/discrete/tsp.py` | Bundled Berlin52, TSPLIB loader, feasible route checks |
| Static knapsack | `src/mmao/discrete/knapsack.py` | Bundled mknap2, OR-Library loader, feasibility and population checks |
| Dynamic continuous | `src/mmao/dynamic/core.py` | Shifted benchmark interfaces and offline error |
| Classification-oriented mixed space | `src/mmao/classification/core.py` | Built-in/custom datasets, selected masks and fitted pipelines |

The modules implement domain-specific resource loops. Their interfaces provide a common software entry point. Exact correspondence of every update equation, configuration, baseline and output table to the journal experiment implementation remains to be audited. The research workspace uses separate study implementations and a pinned 0.2.0 integration example. The version 1.0.1 fixes change some trajectories and classification outputs, so new runs must record the software version.

The scripts in this repository provide package verification and small experiments. Full journal-table reproduction requires the original experiment configurations, reference solver versions, seeds, data and statistical procedures. These have not been imported into this release. The following follow-up work is required for a journal reproduction release.

1. Audit package updates against the archived study code and publish a version-to-equation mapping.
2. Archive exact datasets, seed lists, configurations and solver dependencies.
3. Run all methods under the declared evaluation budgets and stopping rules.
4. Export per-seed results, paired tests, Holm corrections, effect sizes and confidence intervals.
5. Reconcile every generated table against the published results, documenting effects of software fixes.

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
- Cite the IEEE Access article for the published unified-controller study.
- Cite the relevant derivative paper when discussing a specialized domain such as dynamic optimization or classification-oriented search.

## Copyright and license

Copyright (c) 2026 Jinliang Xu.

Author and contact — Jinliang Xu, `jlxufly@gmail.com`, `xujinliang@caict.ac.cn`.

This project is licensed under the Apache License, Version 2.0. Full terms and attribution are available in the LICENSE and NOTICE files.
