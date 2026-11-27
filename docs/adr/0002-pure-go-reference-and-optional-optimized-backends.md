# ADR-0002: Pure Go Reference with Optional Optimized Backends

- Status: Accepted
- Date: 2026-11-27

## Decision

Every supported core operator has a Pure Go correctness path. SIMD and BLAS/LAPACK are optional implementations selected through validated capability and workload dispatch.

## Consequences

- Optimized kernels can be differential-tested.
- Core builds remain portable and do not require CGo/vendor SDKs.
- Kernel registry and dispatch must account for dtype, layout, shape, alignment, and CPU features.
- BLAS threading must be coordinated with GoTorch scheduling to avoid oversubscription.
