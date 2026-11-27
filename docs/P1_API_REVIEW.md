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
