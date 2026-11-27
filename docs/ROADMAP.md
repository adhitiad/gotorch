# GoTorch Roadmap P0–P13

This file is the compact phase overview. **The authoritative task breakdown, dependency graph, Task IDs, status and checklists live in [`TASK.md`](TASK.md).** Do not maintain a separate phase status here.

| Phase | Name | Main deliverable | Dependencies |
|---|---|---|---|
| P0 | Foundation | Module conventions, contracts, errors, platform policy, CI and agent workflow | — |
| P1 | Tensor Engine | Owning tensors, storage, shape/stride, views and tensor data operations | P0 |
| P2 | Automatic Differentiation | Reverse/forward AD, gradients and graph lifecycle | P1 |
| P3 | CPU Compute | Pure Go reference kernels, safe SIMD dispatch and optional BLAS/LAPACK | P1 |
| P4 | Scheduler & Memory | Bounded parallelism, allocator reuse and memory lifetime | P1, P3 |
| P5 | NN & Optimizers | Modules, layers, losses, parameters and optimizers | P2, P3, P4 |
| P6 | Data Engine | Dataset, DataLoader, sampling, transforms and streaming | P0, P4 |
| P7 | Compiler | Typed graph IR, shape inference, optimization, fusion and execution planning | P1, P3, P4 |
| P8 | Training & Inference Runtime | Trainer, sessions, evaluation, checkpoint and resume | P5, P6, P7 |
| P9 | Quantization | Mixed precision, INT8/INT4, calibration and kernels | P3, P5, P8 |
| P10 | Distributed Compute | Workers, collectives, DDP, checkpoint and recovery | P8; contracts begin P0 |
| P11 | Interoperability | ONNX, Safetensors and native model format | P1, P5, P8 |
| P12 | Ecosystem & Tooling | CLI, profiler, registry, extensions, docs and examples | P8–P11 |
| P13 | Production | Security, stress, performance, stability and release | P0–P12 |

## Current implementation checkpoint

The near-term compute emphasis is CPU-first: Pure Go correctness kernels, SIMD acceleration with safe fallback, then optional BLAS/LAPACK. The [Go AI Ecosystem Guide](AI_ECOSYSTEM.md) lists reusable external Go modules without making them GoTorch core dependencies.

P0 contracts exist for errors, dtype, shape, tensor metadata, storage, compute backend, runtime lifecycle, distributed identity and platform policy. The reusable Go packages are being organized under `src/`; see [P0 implementation](P0_IMPLEMENTATION.md).

A phase is not complete because its package or interface exists. It is complete only after its implementation, tests, documentation, platform verification and exit criteria pass. See [`TASK.md`](TASK.md) for the operational checklist.
