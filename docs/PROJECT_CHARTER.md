# GoTorch Project Charter

## Mission
Build a native Go deep-learning and computing framework with a portable correctness baseline, optimized CPU execution, and distributed computing as a first-class architectural capability.

## Problem
Go developers often have to choose between general-purpose Go systems tooling and deep-learning ecosystems centered on Python/C++ runtimes. GoTorch explores whether a Go-native framework can provide a coherent tensor, autodiff, model, training, inference, and distributed-compute stack while preserving Go deployment and operational characteristics.

## Why Go
- Strong static typing and explicit interfaces for maintainable systems APIs.
- Goroutines and channels for bounded concurrency and service integration.
- Simple deployment and a mature cross-platform toolchain.
- A natural fit for production systems where inference and data pipelines coexist with backend services.

Go is not automatically faster than Python or C++; numerical performance depends on kernels, memory layout, vectorization, scheduling, and hardware. Python remains a strong research and ecosystem layer; C/C++ remain important for vendor libraries and low-level kernels. GoTorch should interoperate where it creates measurable value rather than claim universal replacement.

## Product principles
1. Correctness before optimization.
2. Pure Go reference kernels define portable semantics.
3. SIMD and optional BLAS/LAPACK are optimization paths, never correctness prerequisites.
4. Distributed execution is designed in early, but local workloads must not pay unnecessary network overhead.
5. Windows AMD64 is the primary production target; Linux AMD64 and Linux ARM64 are compatibility targets.
6. GPU is an optional future extension, not an early dependency.
7. Public claims require tests, reproducible benchmarks, and documented platform evidence.

## Scope
Long-term scope includes tensors, automatic differentiation, neural networks, optimizers, data loading, training, inference, graph/compiler tooling, quantization, interoperability, profiling, and distributed execution.

## Non-goals
- Claiming to replace PyTorch, TensorFlow, or every Python/C++ workflow.
- Mandatory GPU/CUDA or vendor-specific SDK dependencies in the foundation.
- Publishing performance claims without fair, reproducible methodology.
- Marking roadmap phases complete based only on design documents.

## Current status
The repository is an architecture and foundation scaffold. P0–P13 remain planned until implementation and exit criteria are verified. A checked-in design document is not evidence that its runtime feature exists.

## Success measures
- Correctness and numerical stability across supported dtypes and shapes.
- Predictable memory ownership and bounded concurrency.
- Reproducible CPU benchmarks against documented baselines.
- Reliable builds and tests on supported platforms.
- Useful developer documentation and a sustainable contributor workflow.
