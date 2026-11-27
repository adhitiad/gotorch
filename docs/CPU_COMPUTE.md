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

GoTorch uses `golang.org/x/sys/cpu` as its single runtime detector. `compute.DetectCPUFeatures` snapshots architecture, initialization status, AMD64 AVX2/AES and ARM64 ASIMD/AES. The package accounts for OS-managed extended register state on x86 before reporting AVX2. If initialization is unavailable, consumers must choose the scalar backend. Do not infer support from `GOARCH` alone, and do not execute SIMD merely because a feature flag is present; a kernel must also be compiled for that ISA and pass its own dispatch constraints.

Dispatch must account for architecture, CPU feature availability, OS state support, dtype, layout, shape, alignment and workload size. Small operations may remain scalar to avoid dispatch and scheduling overhead.

## Architecture-isolated SIMD kernels

The first experimental SIMD backend is opt-in at build time with `GOEXPERIMENT=simd` and is isolated by Go build constraints:

- AMD64 uses 256-bit `simd/archsimd` float32x8/float64x4 operations and is selected only when the runtime snapshot reports AVX2 (which also guarantees AVX).
- ARM64 uses 128-bit NEON float32x4/float64x2 operations and is selected only when the runtime snapshot reports ASIMD.
- Other architectures, builds without the experiment, and unsupported CPUs expose no SIMD backend; the default registry always retains the Pure Go reference backend.
- The initial optimized operation set is contiguous float32/float64 add, subtract and multiply. Other operations, strided layouts, unsupported dtypes and workloads smaller than 32 elements stay on the scalar backend.
- The initial storage abstraction requires staging bytes into typed slices before SIMD execution. This prioritizes safe, architecture-isolated correctness; benchmark and storage-mapping work must measure whether the staging cost is worthwhile before widening dispatch.
- SIMD types never appear in GoTorch's public API. The Go standard-library SIMD API is experimental and is not enabled for ordinary builds.

The CI matrix runs both the normal build and a separate `GOEXPERIMENT=simd` build on Windows AMD64, Linux AMD64 and Linux ARM64. Differential tests compare every implemented SIMD operation against the Pure Go reference using `DefaultTolerance`, including a non-vector-aligned tail.
## BLAS / LAPACK

[Gonum](https://github.com/gonum/gonum) is the first Go-native numerical library to evaluate. Its `blas/gonum` package is a Pure Go BLAS implementation with documented incomplete float32/float64 coverage. Its `blas64` and `blas32` packages wrap BLAS implementations.

External OpenBLAS/BLIS/vendor libraries may be integrated behind optional adapters, but must not become mandatory CGo/native dependencies of GoTorch core. Measure conversion, layout and thread-pool overhead, not only the BLAS call.


## Optional OpenBLAS adapter

The optional OpenBLAS backend is compiled only with `-tags=gotorch_blas` on Linux or Windows. Runtime loading is separately opt-in: set `GOTORCH_OPENBLAS_LIBRARY` to the exact shared-library path. The default build has no dynamic-library dependency and does not search the host for BLAS. The adapter currently maps the supported rank-2 `matmul` kernel to row-major CBLAS SGEMM/DGEMM for contiguous float32/float64 operands; it does not claim LAPACK routines because no LAPACK operation is currently part of the compute kernel contract.

The adapter requires the CBLAS GEMM symbols and OpenBLAS thread getter/setter symbols. Missing symbols or an unloadable library produce an explicit constructor error; the registry does not silently select an uncontrolled native backend. Storage is staged through Go-owned typed slices, so the adapter's performance must be benchmarked against scalar and SIMD paths before making speed claims. Cancellation is checked before and after the native call; an in-flight vendor GEMM cannot be interrupted.

### BLAS thread coordination

`ExecutionPolicy` specifies the maximum concurrent GoTorch workers and configured BLAS thread ceiling. The effective BLAS count is `min(BLASThreads, max(1, GOMAXPROCS/GoWorkers))`. The backend serializes each process-global OpenBLAS set/call/restore transaction, preserving the prior thread count and avoiding races between GoTorch callers. This is intentionally conservative until P04 provides a central scheduler/BLAS lease. It does not modify `GOMAXPROCS`. Callers must stop dispatching before closing the backend.

CI compiles and tests the tagged adapter on Windows AMD64, Linux AMD64 and Linux ARM64 without requiring a native OpenBLAS installation; mock CBLAS functions exercise numerical parity and thread-budget behavior. The tagged build is also tested with `CGO_ENABLED=0` so the optional adapter does not introduce a CGo requirement. A real OpenBLAS library integration test remains environment-dependent.

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

### Numerical acceptance policy

The reference backend is deterministic for a fixed input, dtype, operation and build: logical row-major traversal, serial reductions, and float64 intermediate accumulation for both supported dtypes. This does not promise bitwise parity with a parallel reduction or vendor BLAS.

`compute.DefaultTolerance` defines initial differential-test budgets. The comparison is `abs(got-want) <= Abs + Rel*max(abs(got), abs(want))`; NaN matches only NaN, and infinities match only an infinity with the same sign.

| Operation group | float32 (Abs, Rel) | float64 (Abs, Rel) |
|---|---:|---:|
| add/sub/mul/div/neg/relu | 1e-6, 1e-5 | 1e-12, 1e-10 |
| exp/log | 2e-6, 2e-5 | 2e-12, 2e-10 |
| sum/mean | 2e-5, 2e-5 | 2e-11, 2e-10 |
| matmul | 5e-5, 5e-5 | 1e-10, 1e-9 |

These are starting acceptance budgets, not claims of universal numerical error bounds. Tests must include cancellation-prone values, mixed magnitudes, subnormals where supported, signed zero, NaN and infinities. Tighten or revise a budget only with reference evidence and documented rationale.

### Invocation and ownership

A `compute.Invocation` borrows its storage handles for the duration of `Execute`. Backends must not retain the invocation slices or handles. The caller allocates outputs, guarantees they do not overlap writable inputs or one another, and keeps all storage handles alive until execution returns. The reference backend writes only contiguous output metadata and never allocates or retains storage.
