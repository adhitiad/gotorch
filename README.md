# GoTorch

**Go-native universal deep learning and computing framework.**

GoTorch is designed as a full deep-learning framework with a CPU-first hybrid compute engine and first-class distributed-computing architecture.

> Project status: architecture and foundation specification. This repository scaffold does not claim that P0–P13 implementations are complete.

## Fundamental decisions

- **Scope:** full framework — tensors, automatic differentiation, neural networks, optimizers, training, inference, compiler, quantization, and distributed computing.
- **Compute:** hybrid CPU — Pure Go reference kernels, architecture-specific SIMD, and optional BLAS/LAPACK adapters.
- **Distributed:** first-class by design — multi-process and multi-node execution, collectives, distributed training, checkpointing, and recovery.
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
- [`docs/`](docs/): architecture, decisions, roadmap, design contracts, and engineering standards.
- [`.github/workflows/`](.github/workflows/): CI workflows.

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

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for phase deliverables and exit criteria.

## Development status

This is a documentation-first foundation scaffold. No production tensor kernels, autograd engine, compiler, or distributed runtime are claimed to exist yet. Implement each phase behind tested contracts; do not create empty packages merely to match the target tree.

## Go toolchain

The module language baseline is Go 1.23.0 for broad compatibility. CI uses the current Go 1.27 patch line. For development and production, use a currently supported Go toolchain; see the official [Go release history](https://go.dev/doc/devel/release).

## Initial commands

```bash
go test ./...
go test -race ./...
go vet ./...
go test -bench=. -benchmem ./...
```

## License

License selection is intentionally pending. Do not publish or redistribute this project under an assumed license; obtain an explicit project-owner decision first.
