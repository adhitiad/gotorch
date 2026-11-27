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
