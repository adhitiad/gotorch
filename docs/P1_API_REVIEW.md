# P1 Tensor Engine API Review

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

P1 is closed only for P01-001 through P01-011. This decision does not claim that GoTorch is a complete deep-learning framework or that P2/P3 are implemented.
