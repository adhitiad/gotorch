# ADR-0001: Full Framework, Hybrid CPU, Distributed-First Architecture

- Status: Accepted
- Date: 2026-11-27

## Context

GoTorch is intended to become a broad Go-native deep-learning and computing framework rather than an inference-only runtime. The architecture must support local CPU execution and future distributed workloads without making GPU hardware mandatory.

## Decision

1. Build a full framework: tensor, autograd, NN, optimizers, training, inference, compiler, quantization, interoperability, and tooling.
2. Use hybrid CPU compute: Pure Go reference kernels, optional SIMD, optional BLAS/LAPACK.
3. Treat distributed computing as a first-class architectural concern from P0; implement it in a staged manner after local semantics stabilize.
4. Target Windows AMD64 as primary production, with Linux AMD64 and Linux ARM64 compatibility.
5. Keep GPU/vendor SDKs out of core dependencies and early milestones.

## Consequences

- More subsystem contracts must be designed early.
- P0 must explicitly define ownership, placement, execution, and distributed identity boundaries.
- Pure Go reference implementations increase portability and testing leverage.
- Distributed implementation is substantial and must be staged, beginning with multi-process DDP after local training works.
- Performance claims require reproducible benchmarks.
