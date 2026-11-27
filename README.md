# GoTorch

**Go-native deep-learning and computing framework.**

GoTorch is designed as a full deep-learning framework with a CPU-first hybrid compute engine and first-class distributed-computing architecture.

> **Project status:** architecture and foundation scaffold. P0–P13 implementation status must be verified against code, tests, and phase exit criteria; this repository does not claim the full framework is complete.

## Fundamental decisions

- **Scope:** tensors, automatic differentiation, neural networks, optimizers, training, inference, compiler, quantization, interoperability, and distributed computing.
- **Compute:** hybrid CPU — Pure Go reference kernels, architecture-specific SIMD, and optional BLAS/LAPACK adapters.
- **Distributed:** first-class in architecture, without imposing network overhead on local execution.
- **Platforms:** Windows AMD64 primary; Linux AMD64 and Linux ARM64 compatibility.
- **Accelerators:** optional future extensions. GPU is not a core dependency or early implementation priority.
- **Module path:** `github.com/adhitiad/gotorch`.

## Architecture

```text
Public API / Developer Tools
          |
Framework: Tensor · Autograd · NN · Optim · Data · Training
          |
Compiler: Graph IR · Shape Inference · Optimization · Fusion
          |
Unified Runtime: Scheduler · Executor · Memory Planner · Profiler
          |
+-----------------------+--------------------------+
| Local CPU Compute     | Distributed Compute      |
| Pure Go / SIMD / BLAS | Workers / Collectives    |
+-----------------------+--------------------------+
          |
Storage · RAM · Cache · NUMA · Network
```

## Repository guide

- [`AGENTS.md`](AGENTS.md): mandatory instructions for coding agents.
- [`.agent/`](.agent/): agent rules, skills, workflows, and task templates.
- [`docs/`](docs/): architecture, contracts, roadmap, decisions, and engineering standards.
- [`community/`](community/): governance, conduct, support, security, and release policy.
- [`.github/`](.github/): issue forms, pull request template, ownership, dependency updates, and CI workflows.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): contributor workflow.

## Roadmap

| Phase | Scope |
|---|---|
| P0 | Foundation and architecture contracts |
| P1 | Tensor Engine |
| P2 | Automatic Differentiation |
| P3 | CPU Compute and SIMD |
| P4 | Parallel Scheduler and Memory Engine |
| P5 | Neural Networks and Optimizers |
| P6 | Data Engine |
| P7 | Compiler and Graph Optimization |
| P8 | Training and Inference Runtime |
| P9 | Quantization and Numerical Optimization |
| P10 | Distributed Computing |
| P11 | Model Interoperability |
| P12 | Developer Ecosystem and Tooling |
| P13 | Performance, Security, and Production |

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for deliverables and exit criteria. See [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md) for the mission, problem statement, rationale for Go, scope, and non-goals.

## Development status

This is a documentation-first foundation scaffold. Do not infer that production tensor kernels, autograd, compiler, or distributed runtime exist merely because their architecture is documented. Implement phases behind tested contracts and update the roadmap only when acceptance criteria are met.

## Toolchain

The module's declared Go language/toolchain baseline is recorded in [`go.mod`](go.mod). Use a supported Go release compatible with that declaration. CI is the source of truth for the tested version matrix.

## Initial commands

```bash
go test ./...
go test -race ./...
go vet ./...
go test -bench=. -benchmem ./...
```

## Governance and contribution

Read [`community/GOVERNANCE.md`](community/GOVERNANCE.md), [`community/CODE_OF_CONDUCT.md`](community/CODE_OF_CONDUCT.md), [`community/SECURITY.md`](community/SECURITY.md), and [`community/RELEASE_PROCESS.md`](community/RELEASE_PROCESS.md). Questions belong in GitHub Discussions; reproducible defects and proposals belong in the matching issue forms.

## License

GoTorch is licensed under the [MIT License](LICENSE).
