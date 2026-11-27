# GoTorch Architecture

## Product definition

GoTorch is a Go-native full deep-learning and computing framework. It provides tensor computation, automatic differentiation, neural networks, optimization, training, inference, graph compilation, quantization, model interoperability and distributed execution.

## Fundamental architecture decisions

1. Full framework scope; not inference-only.
2. Hybrid CPU compute: Pure Go reference, optional SIMD, optional BLAS/LAPACK.
3. Distributed computing is first-class in contracts and subsystem boundaries.
4. Windows AMD64 is primary production; Linux AMD64 and Linux ARM64 are compatibility targets.
5. GPU and vendor accelerators are optional future extensions, not core dependencies or early priorities.
6. Module path: `github.com/adhitiad/gotorch`.

## Repository package boundaries

- Reusable framework packages live under `src/<package>/`.
- Their import paths use `github.com/adhitiad/gotorch/src/<package>`.
- `cmd/<name>/main.go` owns executable entry points.
- `app/<name>/` owns application-level composition, dependency wiring and external adapters.
- Root `gotorch.go` is a small compatibility facade for the root module import, not a home for framework internals.

## Layers

### Public API and ecosystem
Developer API, CLI, examples, model registry, extensions and tooling.

### Framework
Tensor, autograd, neural-network modules, optimizers, loss, initialization, data, training and inference.

### Compiler and graph
Frontend graph, typed IR, shape inference, graph validation, optimization passes, fusion, memory planning and execution planning.

### Unified runtime
Scheduler, executor, kernel dispatcher, session lifecycle, memory management, profiling and cancellation.

### Local compute
Pure Go kernels, architecture-specific SIMD and optional BLAS/LAPACK adapters.

### Distributed compute
Coordinator, workers, transport, collectives, placement, distributed tensors, checkpointing and recovery.

### Hardware and storage
System RAM, CPU cache, NUMA topology, network and optional accelerator extensions.

## Execution modes

- Eager execution for developer ergonomics and debugging.
- Compiled graph execution for optimized workloads.
- Local parallel execution for one host.
- Distributed execution for multi-process/multi-node workloads.

These modes should share tensor semantics and operator definitions while allowing different execution plans.

## Architectural boundaries

- Framework packages must not depend on concrete SIMD or BLAS implementations.
- Compiler must target runtime/kernel contracts rather than OS-specific details.
- Distributed transport must not leak into ordinary local tensor operations.
- Core module must compile without CGo or vendor SDKs.
- Optional optimized backends must have a reference fallback.

## P0 implementation checkpoint

P0's current foundation contracts are provided by `src/errs`, `src/dtype`, `src/shape`, `src/tensor`, `src/storage`, `src/compute`, `src/runtime`, `src/distributed` and `src/platform`. They do not yet provide a concrete tensor allocator, tensor arithmetic, reference CPU kernels, SIMD/BLAS execution or distributed transport. Those are later-phase deliverables, even though their contracts are defined early.

See [`TASK.md`](TASK.md) for the phase-level task plan and checklists.
