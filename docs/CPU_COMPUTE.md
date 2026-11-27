# CPU Compute Strategy

## Backend hierarchy

1. Pure Go scalar kernels: mandatory portability and reference path.
2. SIMD kernels: optional architecture-specific acceleration.
3. BLAS/LAPACK adapters: optional dense linear algebra acceleration.
4. Specialized kernels: workload-specific implementations selected by measured benefit.

## Dispatch

Dispatch considers CPU capabilities, dtype, layout, shape, alignment, and workload size. Small operations may remain scalar to avoid dispatch and parallel scheduling overhead.

## Parallelism

Use a bounded scheduler and chunked work. Coordinate intra-op parallelism with request-level concurrency and external BLAS thread pools. Never spawn one goroutine per element. Do not modify global `GOMAXPROCS` automatically.

## SIMD portability

Use runtime feature detection. Keep scalar fallback available. Isolate assembly and build-tagged code. Test optimized output against reference. Windows AMD64 and Linux AMD64 are primary SIMD targets; Linux ARM64 requires a separately validated NEON path.

## Numerical behavior

Floating-point reduction order can change results. Define operation-specific tolerances and deterministic modes where required. Do not promise bitwise equality across different kernels unless explicitly guaranteed.
