# P1 Tensor Engine API Review

This document contains two sequential review records. The first covers P01-001 through P01-003; the second covers P01-004 through P01-011.

## P01-001 through P01-003 review

- **Review date:** 2026-11-27
- **Reviewed branch:** `adhitiad_gpt`
- **Reviewed implementation commit:** `b6da83a029607cd004b68ebbbdc5d01c82d9ccf6`
- **Scope:** P01-001 through P01-011.
- **Decision:** approved for P1 closure. The public Tensor/storage contracts, edge cases, ownership behavior, view aliasing and documented limits have been reviewed. No unresolved critical/high-severity blocker remains for the P1 scope.

## Public contract decision

### Ownership and storage
- Each `tensor.Tensor` owns one independently releasable `storage.Storage` handle. Tensor must not be copied after first use; `Retain` and view methods create independently closable handles.
- `Close` is idempotent. Closing one handle does not invalidate other retained tensors or views. The zero value owns no resources; data/metadata methods return `errs.ErrInvalidState`, while `Close` is safe.
- `storage.HeapAllocator` is a zero-initializing Pure Go allocator. Reference-counted allocations are cleared after the final independent handle is released.
- No API returns the backing byte slice or a typed slice alias.

### Constructors, data representation and indexing
- `NewTensor`, `NewTensorWithAllocator`, `From[T]` and `FromWithAllocator[T]` construct owned contiguous tensors.
- `Read[T]` returns a detached row-major copy; `Write[T]` copies values. Dtype must match exactly; there is no implicit numeric conversion.
- Fixed-width values use little-endian storage. Bool values use canonical bytes 0/1; complex values are encoded real component then imaginary component; Float16/BFloat16 remain raw-bit payload types.
- `Metadata.ElementOffset` calculates checked physical element offsets from the metadata strides. `Metadata.LinearIndex` calculates logical row-major indexing independently of physical layout. `At[T]`, `Set[T]` and `Tensor.ByteOffset` provide checked single-element access/offset reporting.

### Views, layout and aliasing
- `Slice` uses an end-exclusive interval and positive step; `Narrow` is unit-step; `Select` removes one axis.
- `Transpose` permutes dimensions and strides. Without axes it reverses the axes; explicit axes map output axes to input axes.
- `Reshape` preserves storage and is allowed only for contiguous metadata with an unchanged element count. It does not silently copy.
- `BroadcastShapes` applies right-aligned broadcast rules. `Metadata.BroadcastTo` represents expanded dimensions with zero strides.
- Every Tensor view retains its backing storage using a distinct handle. Mutations through one non-overlapping alias are observable through the other aliases.
- A zero-stride broadcast view supports reads and single-index `Set`. Bulk `Write` is rejected with `errs.ErrOverlappingWrite` whenever overlap is detected or uniqueness cannot be proven. This prevents ambiguous scatter order.
- `Copy` and `Contiguous` always create an independent row-major allocation, even when the source is already contiguous.

### Binary snapshot
- `MarshalBinary` emits the version-1 `GTNS` format containing magic, version, dtype, rank, dimensions, payload length and logical row-major bytes.
- `UnmarshalBinary` validates magic, supported version, dtype, shape, exact payload length and canonical Bool values before installing a new allocation. It accepts only a zero or closed receiver; failed decoding does not mutate an active receiver.
- The format is versioned but remains pre-1.0. No permanent cross-version checkpoint compatibility guarantee is made.

## Verification

- **Go CI run 45937237974** passed on reviewed implementation commit `b6da83a029607cd004b68ebbbdc5d01c82d9ccf6`.
- Windows AMD64: formatting, package discovery, tests and vet passed. Race tests are skipped by the workflow on Windows.
- Linux AMD64: formatting, package discovery, tests, vet, race tests and Tensor benchmark smoke passed.
- Linux ARM64: formatting, package discovery, tests, vet and race tests passed. The Linux AMD64-only benchmark smoke was skipped as intended.
- The benchmark smoke invokes allocation, view creation, indexing and copy benchmarks with `-benchtime=1x -benchmem`. This is an execution check, not a performance claim.

## Accepted limits and follow-up boundaries

- Negative strides and reverse slicing are not supported.
- `Reshape` requires contiguous input; call `Copy` or `Contiguous` first for a strided input.
- Bulk writes to overlapping/conservatively uncertain layouts are intentionally rejected.
- A multi-element scatter through a custom Storage implementation is not transactional if a custom writer fails partway through. Callers must coordinate conflicting writes through aliases.
- The version-1 binary snapshot is a pre-1.0 serialization contract, not a permanent model-checkpoint interchange guarantee.
- Tensor arithmetic, autograd integration, CPU scalar kernels, SIMD execution and BLAS adapters are not P1 deliverables. CPU compute remains P3 and its current priority is Pure Go reference kernels, then SIMD, then optional BLAS/LAPACK.

<<<<<<< HEAD
P1 is closed only for P01-001 through P01-011. This decision does not claim that GoTorch is a complete deep-learning framework or that P2/P3 are implemented.
=======
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
>>>>>>> origin/main
