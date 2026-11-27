# GoTorch Roadmap P0–P13

Status labels: **Planned** until implementation and exit criteria are verified. This roadmap is not a claim that any phase is complete.

| Phase | Name | Main deliverable | Dependencies |
|---|---|---|---|
| P0 | Foundation | Module, contracts, errors, CI, architecture docs | — |
| P1 | Tensor Engine | Storage, shape/stride, views, dtype, core ops | P0 |
| P2 | Autograd | Reverse/forward AD, gradients, graph lifecycle | P1 |
| P3 | CPU Compute | Pure Go kernels, SIMD dispatch, BLAS adapters | P1 |
| P4 | Scheduler & Memory | Bounded parallelism, allocator, lifetime planner | P1, P3 |
| P5 | NN & Optimizers | Modules, layers, losses, SGD/Adam/AdamW | P2, P3, P4 |
| P6 | Data Engine | Dataset, DataLoader, sampling, transforms, streaming | P0, P4 |
| P7 | Compiler | IR, shape inference, optimization, fusion, planning | P1, P3, P4 |
| P8 | Training & Inference Runtime | Trainer, sessions, evaluation, checkpoint/resume | P5, P6, P7 |
| P9 | Quantization | Mixed precision, INT8/INT4, calibration and kernels | P3, P5, P8 |
| P10 | Distributed Compute | Workers, collectives, DDP, checkpoint/recovery | P8; contracts begin P0 |
| P11 | Interoperability | ONNX, Safetensors, native model format | P1, P5, P8 |
| P12 | Ecosystem & Tooling | CLI, profiler UX, registry, extensions, docs | P8–P11 |
| P13 | Production | Security, stress, performance, stability, release | P0–P12 |

## P0 — Foundation

Deliver: module conventions, errors, platform abstraction, dtype/shape metadata contract, storage ownership contract, compute interfaces, runtime lifecycle, distributed identity contracts, test infrastructure, CI matrix, architecture documentation.

Exit: core builds on Windows AMD64 and Linux AMD64; Linux ARM64 native test or documented cross-build; contracts reviewed; no mandatory CGo/vendor SDK.

## P1 — Tensor Engine

Deliver: typed storage, shape/stride, broadcasting, views, slicing, reshape, transpose, contiguous conversion, arithmetic, serialization primitives.

Exit: correctness, overflow, aliasing, ownership, zero-size and non-contiguous tests pass.

## P2 — Autograd

Deliver: reverse-mode AD, forward-mode foundations, gradient accumulation, backward graph, custom gradient hooks, gradient checking.

Exit: finite-difference gradient checks and lifecycle/concurrency tests pass.

## P3 — CPU Compute

Deliver: Pure Go reference kernels, dispatch registry, CPU feature detection, SIMD implementations, optional BLAS/LAPACK, optimized MatMul/reductions.

Exit: differential tests pass; safe feature dispatch; reproducible benchmarks.

## P4 — Scheduler and Memory

Deliver: bounded worker scheduler, chunking, cancellation, arenas, buffer reuse, memory statistics, execution lifetime.

Exit: race/deadlock/leak tests pass; oversubscription bounded; memory ownership documented.

## P5 — Neural Networks and Optimizers

Deliver: Module, parameters, Linear, activations, normalization, convolution foundations, loss, SGD, Adam, AdamW, LR scheduler.

Exit: forward/backward correctness and XOR/regression training examples pass.

## P6 — Data Engine

Deliver: Dataset, DataLoader, sampler, batching, shuffle, transforms, prefetch, streaming, caching.

Exit: deterministic seeding, cancellation, worker lifecycle, and throughput tests pass.

## P7 — Compiler and Graph Optimization

Deliver: graph IR, shape inference, validation, constant folding, dead-code elimination, fusion, execution/memory planner.

Exit: optimized graph matches reference within documented tolerances; compile and run costs measured separately.

## P8 — Training and Inference Runtime

Deliver: Trainer, evaluation, inference session, batching, checkpoint, resume, profiling.

Exit: train-save-load-infer workflow works; session concurrency and checkpoint consistency tested.

## P9 — Quantization

Deliver: FP16/BF16 where practical, INT8/INT4 paths, calibration, quantized kernels, mixed precision.

Exit: accuracy and performance trade-offs measured; no silent conversion.

## P10 — Distributed Computing

Deliver: coordinator, worker, transport, collectives, DDP, parameter synchronization, distributed checkpoint, recovery.

Exit: multi-process training, collective correctness, worker failure, timeout, cancellation, and recovery tests pass.

## P11 — Interoperability

Deliver: ONNX import/export for declared operator coverage, Safetensors, versioned native model format.

Exit: compatibility matrix, malformed-file tests, parity tests, and resource limits.

## P12 — Ecosystem and Tooling

Deliver: CLI, model inspection, profiler tooling, examples, registry/extension contracts, tutorials.

Exit: developer workflows documented and integration-tested.

## P13 — Production

Deliver: API stabilization, fuzzing, security review, stress tests, performance baselines, cross-platform release automation.

Exit: all release gates pass; known limitations and reproducible benchmark reports published.
