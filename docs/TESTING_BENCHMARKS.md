# Testing and Benchmarking

## Test layers

- Unit tests: package semantics and edge cases.
- Property/fuzz tests: shape math, serialization, graph validation, parsers.
- Differential tests: optimized kernels against Pure Go reference.
- Integration tests: model training/inference and data pipeline.
- Concurrency tests: race, cancellation, shutdown, boundedness.
- Distributed tests: multi-process collectives, worker loss, timeout, recovery.

## Numerical validation

Use operation-specific absolute and relative tolerances. Account for floating-point reduction order and precision changes. Record determinism guarantees.

## Benchmark dimensions

- Tensor shapes: small, medium, large, contiguous and non-contiguous.
- Dtypes and layouts.
- Worker counts and request concurrency.
- CPU features and BLAS thread configuration.
- Cold start versus steady state.
- Allocations, bytes/op, peak memory.
- P50/P95/P99, throughput, and scaling efficiency.

## Fair comparison

Record CPU model, OS, Go version, compiler flags, thread counts, model, batch, precision, warmup, repetitions, and baseline configuration. Never compare different workloads and call the result a framework speedup.
