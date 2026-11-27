# GoTorch

**Go-native deep-learning and numerical-computing framework.**

GoTorch is designed as a full deep-learning framework with a CPU-first hybrid compute engine and first-class distributed-computing architecture.

> **Project status:** P0 foundation contracts and the `src/` source-layout migration have passed native CI on Windows AMD64, Linux AMD64 and Linux ARM64. Maintainer review of public contracts remains before P0 closure. The full Tensor Engine, autograd system, CPU kernel library, and distributed runtime are not yet complete. See [the P0 checkpoint](docs/P0_IMPLEMENTATION.md) and [the canonical P0–P13 task plan](docs/TASK.md).

## Fundamental decisions

- **Scope:** tensors, automatic differentiation, neural networks, optimizers, data pipelines, training, inference, compiler, quantization, interoperability and distributed computing.
- **Compute:** hybrid CPU — Pure Go reference kernels, architecture-specific SIMD and optional BLAS/LAPACK adapters.
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

## Repository layout

- `src/`: reusable Go framework packages and tests.
- `cmd/`: executable entry points (`package main`).
- `app/`: application composition, adapters and dependency wiring.
- `gotorch.go`: small root public API facade.
- `docs/TASK.md`: canonical roadmap, P0–P13 descriptions, dependency map and implementation checklists.
- `AGENTS.md`: shared instructions for coding agents.
- `docs/AGENT_SETUP.md`: discovery/setup guidance for Codex, OpenClaw, OpenCode, Hermes and Copilot.
- `.agents/skills/`: portable reusable agent skills.
- `.agent/rules/`, `.agent/workflows/`, `.agent/templates/`: engineering rules and task playbooks.
- `community/`: governance, conduct, support, security and release policy.
- `.github/`: issue forms, PR template, CODEOWNERS, dependency updates and CI.

## Roadmap

| Phase | Scope |
|---|---|
| P0 | Foundation and architecture contracts |
| P1 | Tensor Engine |
| P2 | Automatic Differentiation |
| P3 | CPU Compute and SIMD |
| P4 | Scheduler and Memory Engine |
| P5 | Neural Networks and Optimizers |
| P6 | Data Engine |
| P7 | Compiler and Graph Optimization |
| P8 | Training and Inference Runtime |
| P9 | Quantization |
| P10 | Distributed Computing |
| P11 | Model Interoperability |
| P12 | Developer Ecosystem and Tooling |
| P13 | Production Readiness |

Read [`docs/TASK.md`](docs/TASK.md) for phase descriptions, task IDs, dependencies, checklists and agent handoff protocol. [`docs/ROADMAP.md`](docs/ROADMAP.md) provides the compact phase overview. [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md) explains the project mission, problem statement, rationale for Go, scope and non-goals.

## Agent interoperability

Use the repository-level `AGENTS.md` as the common instruction entry point. It points every agent to the same `docs/TASK.md`, rules and portable skills. Do not maintain separate copies of task status for individual agent products.

## Development status

P0 currently establishes typed metadata, error, storage ownership, compute backend, runtime lifecycle, platform and distributed identity contracts. It does not implement tensor arithmetic, concrete memory allocation, reference CPU kernels, SIMD/BLAS execution or distributed transport. A production capability exists only after its implementation, tests and phase exit criteria pass.

## Toolchain and commands

The module's declared Go version is recorded in `go.mod`. Use a compatible supported release.

```bash
go test ./...
go test -race ./...
go vet ./...
go test -bench=. -benchmem ./...
```

## Governance and contribution

Read [`CONTRIBUTING.md`](CONTRIBUTING.md), [`community/GOVERNANCE.md`](community/GOVERNANCE.md), [`community/CODE_OF_CONDUCT.md`](community/CODE_OF_CONDUCT.md), [`community/SECURITY.md`](community/SECURITY.md) and [`community/RELEASE_PROCESS.md`](community/RELEASE_PROCESS.md).

## License

GoTorch is licensed under the [MIT License](LICENSE).
