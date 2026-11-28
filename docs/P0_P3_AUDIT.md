# P0–P3 Engineering Audit

- **Audit date:** 2026-11-28
- **Reviewed branch:** `adhitiad_gpt` (fixes prepared on `fix/p0-p3-integrity-audit`)
- **Scope:** P0 Foundation, P1 Tensor Engine, P2 Automatic Differentiation and P3 CPU Compute.
- **Review standard:** implementation and executable tests take precedence over completed checklist markers or historical CI runs.

## Executive result

P0–P3 provide a substantial, coherent Go-native foundation, but the audit found two correctness gaps and one documentation drift issue that warranted code changes. The fixes are included in this audit branch and require the current CI matrix before being considered verified.

| Phase | Audit result | Action |
|---|---|---|
| P0 Foundation | Core contracts are present; README and root package status text understated implemented functionality | Corrected status text; no change to the module version |
| P1 Tensor Engine | Mutations through the same Tensor handle used read locks, allowing concurrent writes to interleave with custom Storage implementations | Changed write paths to exclusive handle locks; added a concurrent-write regression using deliberately unsynchronized test storage |
| P2 Autograd | Public context-taking APIs used plain `ctx == nil`, which misses typed-nil context implementations and can panic on `ctx.Err()` | Added typed-nil detection to Backward, JVP and CheckGradient; added regression coverage for Backward/JVP |
| P3 CPU Compute | Reference, SIMD and optional OpenBLAS paths exist with differential tests and benchmarks; scope and limitations are documented | Retained scalar fallback; prior follow-up adds Registry.Close and SIMD threshold sweep; no threshold retuning from a single CI host |

## P0 — Foundation

### Confirmed

- Module and package layout, typed errors, dtype/shape metadata, storage ownership, kernel invocation, runtime lifecycle, platform support and distributed identity contracts are implemented.
- The distributed package currently defines identity/membership validation only; it does not implement transport, authentication or collectives. Those belong to P10 and must not be described as delivered distributed execution.
- Core remains Go-native; optional BLAS is build/runtime opt-in and the scalar CPU path does not require CGo or a vendor SDK.

### Findings and changes

- **Documentation drift:** `README.md` described P3 CPU detection, SIMD, BLAS and benchmarks as still open, and `gotorch.go` described tensor/autograd/CPU kernels as still under development. This contradicted the current branch implementation. Both status descriptions now reflect P0–P3 accurately and retain the pre-1.0 warning.

## P1 — Tensor Engine

### Confirmed

- Tensor handles own independently retained storage references; metadata is immutable through exported accessors.
- Shape/stride calculations, view creation, aliasing rules, checked offsets, serialization and bounds checks have dedicated tests.
- Bulk writes reject overlapping layouts. Negative strides remain unsupported; custom-storage strided writes are not transactional.

### Finding and fix

- **Correctness — concurrent writes through one handle:** `Tensor.writeAll` and `Tensor.writeElementBytes` used `RLock` while mutating storage. Two goroutines writing the same Tensor could therefore enter the write path concurrently. Heap storage happens to serialize its own operations, but the Storage contract does not promise arbitrary overlapping writes are serialized by every implementation. Both mutation paths now take the Tensor's exclusive lock. A regression test uses a deliberately non-locking test Storage to expose the contract violation under `go test -race`.

- **Remaining concurrency boundary:** distinct retained handles and views have independent Tensor locks. Callers must synchronize overlapping writes across those aliases. This is now stated in the Tensor API documentation and type contract.

## P2 — Automatic Differentiation

### Confirmed

- Tape owns forward snapshots and gradient buffers; reverse-mode accumulation is staged before commit.
- Custom VJP/JVP callbacks receive owned snapshots and callback results are validated.
- Graph mutation/close is blocked while backward/JVP is active; cancellation and repeated backward behavior have regression coverage.
- Built-in operations remain deliberately narrow: float32/float64 Add, Mul and Sum; no broadcast-gradient reduction, matmul gradient, higher-order differentiation or parallel backward scheduler.

### Finding and fix

- **Correctness — typed-nil contexts:** `ctx == nil` does not detect an interface containing a nil pointer. Calling `ctx.Err()` can panic. A typed-nil-aware context check is now shared by Backward, JVP and CheckGradient. Regression tests exercise Backward and JVP.

## P3 — CPU Compute

### Confirmed

- Pure Go reference kernels support float32/float64 elementwise operations, reductions and rank-2 matmul, including strided inputs.
- SIMD is opt-in with `GOEXPERIMENT=simd`; AMD64 dispatch requires AVX2 and ARM64 dispatch requires ASIMD. Unsupported builds/CPUs retain the scalar path.
- OpenBLAS is opt-in via `gotorch_blas` and an explicit library path; current adapter scope is contiguous float32/float64 rank-2 CBLAS GEMM.
- Numerical comparison has operation/dtype-specific tolerances and special-value handling.
- Registry shutdown closes registered closable backends; the SIMD threshold benchmark sweeps both sides of the current 32-element cutoff.

### Remaining limitations (not P0–P3 blockers)

- Scalar kernels perform per-element storage access and allocate temporary buffers; benchmark results identify this as a high-priority P4 memory/workspace optimization.
- OpenBLAS process-global thread setting transactions are serialized. Go workers do not execute native GEMM concurrently while that process-global setting lock is held.
- SIMD and BLAS benchmark ratios are host-specific and include staging/storage overhead; they are not isolated ISA speedup claims.
- LAPACK, broadcasting, batched matmul, low-precision/complex compute and additional kernels remain unsupported.

## Verification required for this audit branch

The code changes must pass:
- `gofmt` and `go test ./...`
- `go test -race ./...` on Linux AMD64 and Linux ARM64
- `go vet ./...`
- normal, SIMD-experiment and tagged OpenBLAS CI paths on Windows AMD64, Linux AMD64 and Linux ARM64 as configured
- documentation validation

Do not mark the fixes verified until the current audit commit's workflow runs complete successfully. Historical runs validate their respective historical commits only.
