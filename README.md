# GoTorch

**Go-native deep-learning and numerical-computing framework.**

GoTorch is designed as a full deep-learning framework with a CPU-first hybrid compute engine and first-class distributed-computing architecture.

> **Project status:** P0 Foundation, P1 Tensor Engine and P2 Automatic Differentiation are implemented and validated. P2 adds a Go-native tape/graph engine, reverse-mode VJP, forward-mode JVP, gradient accumulation, custom differentiation rules and finite-difference gradient checks for float32/float64. P3 is in progress: its first slice adds a deterministic backend registry and Pure Go float32/float64 reference kernels for elementwise math, reductions and rank-2 matrix multiplication. CPU feature detection, SIMD dispatch, optional BLAS/LAPACK and optimized-kernel benchmarks remain open, as do the later training runtime, quantization and distributed phases. This is still an early pre-1.0 framework, not a production-ready full deep-learning framework. See the [Autograd API](docs/AUTOGRAD_API.md), [P1 API review](docs/P1_API_REVIEW.md), [Tensor API](docs/TENSOR_API.md), and [canonical P0–P13 task plan](docs/TASK.md).

## Go AI ecosystem guide

New contributors should start with **[Go AI Ecosystem Guide](docs/AI_ECOSYSTEM.md)**. It maps real Go modules for training, Hugging Face downloads/tokenizers, numerical computing, CPU feature detection, SIMD, BLAS and dataset formats.

Important distinction: GoTorch is its own framework. Projects such as [GoMLX](https://github.com/gomlx/gomlx) and [go-huggingface](https://github.com/gomlx/go-huggingface) already provide useful Go-native capabilities, but their features are not automatically GoTorch features or dependencies.

**Current GoTorch engineering priority: Pure Go CPU reference kernels first, SIMD acceleration second, optional BLAS/LAPACK third.** GPU and native accelerator runtimes are not core requirements.

## Module and package documentation

- **Go module:** `github.com/adhitiad/gotorch`
- **Documentation:** [pkg.go.dev/github.com/adhitiad/gotorch](https://pkg.go.dev/github.com/adhitiad/gotorch)
- **First preview target:** `v0.1.0-alpha.1`

The first public preview is intentionally versioned before `v1.0.0`. Exported APIs may change while the framework is developed; this release does not promise backward compatibility.

After the `v0.1.0-alpha.1` tag is published, install the module with:

```bash
go get github.com/adhitiad/gotorch@v0.1.0-alpha.1
```

The root module provides a small compatibility facade. Reusable framework APIs are organized under `src/`, for example:

```go
import (
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/tensor"
)
```

See [docs/PUBLISHING.md](docs/PUBLISHING.md) for the Go module release and pkg.go.dev indexing procedure.

## Fundamental decisions

- **Scope:** tensors, automatic differentiation, neural networks, optimizers, data pipelines, training, inference, compiler, quantization, interoperability and distributed computing.
- **Compute:** CPU-first hybrid — Pure Go reference kernels, Go SIMD/architecture-specific acceleration and optional BLAS/LAPACK adapters.
- **Distributed:** first-class in architecture, without imposing network overhead on local execution.
- **Platforms:** Windows AMD64 primary; Linux AMD64 and Linux ARM64 compatibility.
- **Accelerators:** optional future extensions. GPU is not a core dependency or early implementation priority.

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
- `gotorch.go`: small public root-package facade.
- `docs/TASK.md`: canonical roadmap, P0–P13 descriptions, dependencies and task checklists.
- `docs/AI_ECOSYSTEM.md`: Go libraries and recommended AI development paths.
- `AGENTS.md`: shared instructions for coding agents.
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

## Toolchain and commands

Use the Go version declared in `go.mod`.

```bash
go test ./...
go test -race ./...
go vet ./...
go test -bench=. -benchmem ./...
```

See [`docs/CPU_COMPUTE.md`](docs/CPU_COMPUTE.md) for CPU/SIMD design and [`docs/AI_ECOSYSTEM.md`](docs/AI_ECOSYSTEM.md) for optional modules.

## Governance and contribution

Read [`CONTRIBUTING.md`](CONTRIBUTING.md), [`community/GOVERNANCE.md`](community/GOVERNANCE.md), [`community/CODE_OF_CONDUCT.md`](community/CODE_OF_CONDUCT.md), [`community/SECURITY.md`](community/SECURITY.md) and [`community/RELEASE_PROCESS.md`](community/RELEASE_PROCESS.md).

## License

GoTorch is licensed under the [MIT License](LICENSE).
