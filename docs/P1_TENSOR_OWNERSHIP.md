# P1 Tensor Ownership and Heap Storage — Initial Implementation

> Implementation and maintainer review for P01-001 through P01-003 are complete on `adhitiad_gpt`. See [P1 API Review](P1_API_REVIEW.md) and the [successful Go CI run](https://github.com/adhitiad/gotorch/actions/runs/45935038962).

## Scope

This implementation covers P1-001 through P1-003 only. It does not claim that slicing, views, reshape, transpose, broadcasting, indexing, copy/contiguous operations, or tensor serialization are complete.

## Ownership decisions

- An owning `tensor.Tensor` wraps exactly one independently releasable `storage.Storage` handle.
- Do not copy a Tensor after first use. Use `Tensor.Retain` for another handle sharing the same allocation.
- `Tensor.Close` is idempotent. It releases only the Tensor's handle; another retained Tensor remains valid.
- The zero Tensor value owns no memory. Its metadata and data operations return `errs.ErrInvalidState`; calling Close on it is safe and idempotent.
- A constructor returns one owning handle. On a construction or typed-write error after allocation, it closes the partially created Tensor.
- Storage is accessed only through bounded ReaderAt/WriterAt calls. No backing byte or typed slice is exposed.

## Storage decisions

- `storage.HeapAllocator` is a dependency-free Pure Go, zero-initializing allocator.
- Every Retain returns a distinct handle. Each handle has its own released state; the shared allocation is reclaimed after the final Release.
- Handle lifecycle and allocation bytes are synchronized. Concurrent reads are safe; writes are serialized by the allocation lock. Higher layers should still avoid conflicting logical writes.
- A zero-byte allocation is valid. Negative sizes, nil/typed-nil contexts, cancelled contexts and sizes not representable as a Go slice length are rejected before allocation.
- ReadAt and WriteAt follow standard partial-I/O semantics; they copy bytes and never return an alias to backing memory.

## Typed data API

- `tensor.From[T]` and `tensor.FromWithAllocator[T]` construct tensors from supported fixed-width Go element types.
- `tensor.Read[T]` returns a detached copy.
- `tensor.Write[T]` copies caller values into storage and requires exact dtype and element-count matches. Numeric conversion is never implicit.
- The in-memory typed representation is little-endian. Bool uses canonical byte values 0 and 1. Complex values are stored as real then imaginary components. Float16/BFloat16 are exposed as raw bit payload types until conversion semantics are implemented.
- Current typed access supports contiguous tensors only; future strided/view APIs must implement gather/scatter semantics before this restriction is relaxed.

## Deferred work

P1-004 onward owns checked indexing, views and aliasing rules, slicing, reshape/transpose, broadcasting, contiguous conversion, property testing, performance benchmarks, and complete user-facing examples. The current API intentionally does not expose mutable views.
