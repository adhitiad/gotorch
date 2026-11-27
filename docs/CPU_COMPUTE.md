# CPU Compute Strategy

## Current priority

**CPU + SIMD is the immediate compute direction for GoTorch.** Do not make GPU, CUDA, CGo or vendor SDKs prerequisites for the core framework.

The mandatory baseline is a Pure Go scalar implementation. SIMD and BLAS are optional acceleration paths selected only after correctness tests and reproducible benchmarks.

## Backend hierarchy

1. Pure Go scalar kernels: mandatory portability and reference path.
2. Go SIMD kernels: portable SIMD where possible, architecture-specific paths where justified.
3. BLAS/LAPACK adapters: optional dense linear algebra acceleration.
4. Specialized fused kernels: workload-specific implementations selected by measured benefit.

## Go 1.27 SIMD

Go 1.27 provides experimental `simd` and `simd/archsimd` APIs, enabled with `GOEXPERIMENT=simd`. The API is not stable yet.

- Prefer `simd` for portable, vector-size-agnostic code.
- Use `simd/archsimd` only for architecture-specific operations that materially improve a kernel.
- Keep SIMD implementation behind internal dispatch/build constraints.
- Do not expose experimental SIMD types in GoTorch's public API.
- Always retain the scalar fallback and compare optimized output against it.
- Keep normal build/test jobs without the experiment; add separate experimental SIMD verification.

Reference: [Go 1.27 SIMD release notes](https://go.dev/doc/go1.27), [portable SIMD design](https://go.dev/blog/simd-experiment), [architecture-specific SIMD](https://go.dev/blog/archsimd).

## Runtime CPU feature detection

Use runtime feature detection rather than compile-time assumptions. The optional module `golang.org/x/sys/cpu` exposes architecture feature flags such as AMD64 AVX2/AVX-512 and ARM64 ASIMD. Optional alternatives such as `github.com/klauspost/cpuid/v2` provide broader CPU/topology detail; avoid duplicate feature-detection dependencies unless a measured requirement justifies them.

Dispatch must account for architecture, CPU feature availability, OS state support, dtype, layout, shape, alignment and workload size. Small operations may remain scalar to avoid dispatch and scheduling overhead.

## BLAS / LAPACK

[Gonum](https://github.com/gonum/gonum) is the first Go-native numerical library to evaluate. Its `blas/gonum` package is a Pure Go BLAS implementation with documented incomplete float32/float64 coverage. Its `blas64` and `blas32` packages wrap BLAS implementations.

External OpenBLAS/BLIS/vendor libraries may be integrated behind optional adapters, but must not become mandatory CGo/native dependencies of GoTorch core. Measure conversion, layout and thread-pool overhead, not only the BLAS call.

## Dispatch

Dispatch considers CPU capabilities, dtype, layout, shape, alignment, and workload size. Each optimized kernel must declare what it supports and must have a tested scalar equivalent.

## Parallelism

Use a bounded scheduler and chunked work. Coordinate intra-op parallelism with request-level concurrency and external BLAS thread pools. Never spawn one goroutine per element. Do not modify global `GOMAXPROCS` automatically.

## Portability

- Windows AMD64: primary target; retain scalar and validated SIMD paths.
- Linux AMD64: compatibility target; retain scalar and validated SIMD paths.
- Linux ARM64: compatibility target; add and test an explicit NEON/ASIMD path.
- An unsupported CPU feature must select the fallback, never cause an illegal instruction.

## Numerical behavior

Floating-point reduction order can change results. Define operation-specific tolerances and deterministic modes where required. Do not promise bitwise equality across different kernels unless explicitly guaranteed.

## Benchmarks

Benchmarks must report shape, dtype, layout, architecture, Go version, build flags (including whether `GOEXPERIMENT=simd` is enabled), allocations and throughput. Compare scalar and optimized paths using the same workload and verify output before reporting speedup.


## P3 reference kernel implementation

The first Pure Go reference backend is registered as `compute.ReferenceCPUBackend` and selected through `compute.Registry`. Registration order is the deterministic dispatch precedence: optimized backends are registered before the scalar fallback.

The initial reference operation set is:

- Elementwise binary: `add`, `sub`, `mul`, `div` (identical shapes).
- Elementwise unary: `neg`, `exp`, `log`, `relu`.
- Full reductions: `sum`, `mean`.
- Rank-2 matrix multiplication: `matmul`.

The reference path supports float32 and float64. Inputs may use non-negative strided layouts; outputs must be contiguous. Integer, complex, float16 and bfloat16 arithmetic, broadcasting, batched matrix multiplication and partial-axis reductions are not implemented in this slice.

### Numerical and cancellation policy

- Element traversal and reductions use deterministic logical row-major order.
- Float32 inputs are decoded exactly to float64 for scalar calculation and rounded once when stored as float32. Float64 operations retain float64 precision.
- Reductions and matrix products accumulate in float64 for both supported dtypes, then cast once to the output dtype. This is an explicit first-version accuracy policy, not a promise of bitwise parity with vendor BLAS.
- Division follows IEEE-754 behavior: zero divisors may produce infinities or NaNs. Logarithms of zero/negative inputs follow Go's `math.Log` behavior. ReLU propagates NaN.
- Cancellation is checked during long elementwise loops, reductions and matrix products. A cancelled operation may have partially written its caller-provided output; callers must discard or overwrite that output after an error.
- Different optimized implementations may use different reduction order. They must declare their numerical behavior and pass operation-specific differential tolerances before dispatch is enabled.

### Invocation and ownership

A `compute.Invocation` borrows its storage handles for the duration of `Execute`. Backends must not retain the invocation slices or handles. The caller allocates outputs, guarantees they do not overlap writable inputs or one another, and keeps all storage handles alive until execution returns. The reference backend writes only contiguous output metadata and never allocates or retains storage.
