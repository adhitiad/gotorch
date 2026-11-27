# P1 Tensor API and Ownership Review

- **Review date:** 2026-11-27
- **Reviewed branch:** `adhitiad_gpt`
- **Scope:** P01-001, P01-002 and P01-003 only.
- **Reviewed implementation commit:** `4e77ab05fd6668f638945e6d5d99edfd16061c34`
- **Decision:** approved for completion of P01-001 through P01-003 after the native CI run linked below.

## Public contract decision

### Tensor ownership and lifecycle

- `tensor.Tensor` is an owning handle to exactly one independently releasable `storage.Storage` handle.
- Tensor values must not be copied after first use. `Tensor.Retain` creates a second Tensor handle and increments the underlying allocation reference count.
- Each retained Tensor has an independent close lifecycle. Closing one handle does not invalidate the other retained handles.
- `Tensor.Close` is idempotent. It releases only the current handle.
- The zero value owns no storage. `Metadata`, `Retain`, `Read` and `Write` reject it with `errs.ErrInvalidState`; `Close` is safe and idempotent.
- A nil Tensor receiver is treated as an invalid argument by methods that return errors; `Valid` returns false.
- Tensor metadata and storage references remain private. No API exposes an internal byte slice or typed slice.

### Pure Go heap allocator

- `storage.HeapAllocator` is dependency-free and its zero value is usable.
- Allocation returns zero-initialized heap memory; zero-byte allocations are valid.
- Negative sizes, typed-nil contexts, cancelled contexts and sizes not representable as a Go slice length are rejected.
- Each `Retain` returns a distinct handle with its own released state.
- Releasing one handle does not invalidate other retained handles. The allocation bytes are cleared and discarded only after the final handle is released.
- Reads and writes follow `io.ReaderAt`/`io.WriterAt` partial-I/O conventions. The allocation lock serializes writes and coordinates reads; concurrent retained-handle writes to disjoint offsets are exercised by a race test.

### Constructors and typed data

- `NewTensor` allocates zero-initialized contiguous storage through the default Pure Go allocator.
- `NewTensorWithAllocator` permits an allocator to be injected while retaining ownership of the returned handle.
- `From[T]` and `FromWithAllocator[T]` create a Tensor and copy the input values.
- `Read[T]` returns a detached copy. `Write[T]` copies into the Tensor.
- Dtype and element count must match exactly; implicit numeric conversion is not performed.
- Fixed-width scalar types use a little-endian representation. Bool values are canonicalized to bytes 0 and 1. Complex values store real then imaginary components. `Float16Bits` and `BFloat16Bits` preserve raw 16-bit payloads; numerical conversion helpers are not part of this task.
- The generic element set intentionally omits machine-sized `int` and `uint` so that representation width is explicit.

## Verification and limitations

The following GitHub Actions run passed on the reviewed implementation commit:

- [Go CI run 45935038962](https://github.com/adhitiad/gotorch/actions/runs/45935038962)
- Windows AMD64: formatting, package discovery, tests and vet passed; race testing is skipped by the workflow on Windows.
- Linux AMD64: formatting, package discovery, tests, vet and race testing passed.
- Linux ARM64: formatting, package discovery, tests, vet and race testing passed.

The race suite includes concurrent disjoint writes through independently retained storage handles. The local environment used during development was Go 1.23.2, while the repository declares Go 1.27.4; the native Go 1.27 CI matrix is the authoritative verification.

The following remain outside this review and are not claimed implemented:

- Row-major multidimensional indexing and checked offsets (`P01-004`).
- Views, shared writable aliasing rules, slicing, narrow/select, reshape and transpose (`P01-005`, `P01-006`).
- Broadcasting, contiguous conversion, and explicit copy APIs (`P01-007`, `P01-008`).
- Property tests, benchmark suite and complete public examples (`P01-009` through `P01-011`).
- Typed access currently accepts contiguous tensors only. Float16/BFloat16 conversion semantics remain future work.

This approval completes only P01-001 through P01-003. It does not close P1 as a phase.

## P01-004 through P01-011 completion review

- **Review date:** 2026-11-27
- **Reviewed implementation:** `7ea30fe628709e325b646bedb29f1236e926efb7`, merged into `main` as `52b463e6c151c0263bd86c15c01e57453a9412bc`.
- **Follow-up fix:** `6cb28b8e1b9d5c801f479513ad6fa2ccf41e420e` (empty views at the end of a positive-step slice, plus scalar/empty layout coverage).
- **Decision:** approved for completion of P01-004 through P01-011 after the native CI run linked below.

### Row-major indexing and metadata operations

- `Metadata.ElementOffset` applies element strides with checked multiplication and addition. Invalid ranks and coordinates return `errs.ErrOutOfBounds`; arithmetic overflow returns `errs.ErrShapeOverflow`.
- `Metadata.LinearIndex` computes the row-major logical index independently of physical strides.
- `Tensor.ByteOffset`, `At[T]` and `Set[T]` use the same checked offset path. `At` and `Set` require an exact dtype match and work through contiguous, sliced, transposed and broadcast views.
- `Metadata.Slice` is end-exclusive with `step > 0`; `Narrow` is the unit-step form; `Select` removes one axis; `Transpose` permutes shape and strides; `Reshape` requires contiguous metadata and preserves element count.
- `BroadcastShapes` right-aligns dimensions and accepts equal dimensions, expansion from one, and zero only against zero or one. `Metadata.BroadcastTo` and `Tensor.BroadcastTo` represent expanded axes with zero strides.
- Empty views are supported at and past the source's last reachable element. They have no addressable element and never access storage.

### Views, aliasing and copy

- Every view retains the shared allocation as a separate storage handle, so closing the source does not invalidate an existing view and closing a view does not close the source.
- Writes through any alias are visible through the other aliases. `Set` is deterministic for one explicit logical index, including broadcast aliases.
- Bulk `Write` rejects overlapping or conservatively uncertain layouts with `errs.ErrOverlappingWrite` before writing. Non-overlapping strided layouts scatter in logical row-major order.
- `Copy` and `Contiguous` always allocate independent contiguous storage, even for contiguous sources.
- Scalar (rank zero) and empty tensors work with indexing, views, broadcasting and copy.

### Verification

The native Go 1.27 matrix passed on `52b463e6c151c0263bd86c15c01e57453a9412bc`:

- [Go CI run 45937063416](https://github.com/adhitiad/gotorch/actions/runs/45937063416): Windows AMD64, Linux AMD64 and Linux ARM64 tests and vet passed; Linux AMD64 and Linux ARM64 also ran `go test -race ./...`.
- [Go CI run 45937063373](https://github.com/adhitiad/gotorch/actions/runs/45937063373): repository validation passed.

The follow-up fix and edge-case tests are verified by [Go CI run 45939252569](https://github.com/adhitiad/gotorch/actions/runs/45939252569) on `6cb28b8e1b9d5c801f479513ad6fa2ccf41e420e`.

Local verification on the follow-up commit: `gofmt -l .` produced no output; `go test ./...`, `go vet ./...` and the tensor benchmarks passed. The local Windows environment has no CGO C compiler, so the race detector was not run locally; the Linux CI jobs remain authoritative.

### Limitations

- Negative strides, advanced indexing, in-place arithmetic and multi-view mutation transactions remain future work.
- The overlapping-layout detector is conservative: some provably non-overlapping permutations may still be rejected for bulk `Write`.
- Multi-element strided writes are not transactional when a custom Storage implementation fails partway through.
- This review completes P01-004 through P01-011 and the P1 checklist. It does not claim tensor arithmetic, CPU kernels, autograd or checkpoint stability.
