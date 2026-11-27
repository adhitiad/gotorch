# P1 Tensor Engine — Ownership, Views and Data Semantics

> P01-001 through P01-011 are implemented and reviewed. The empty-view follow-up passed the native Go CI matrix in [Go CI run 45939341927](https://github.com/adhitiad/gotorch/actions/runs/45939341927); see [P1 API Review](P1_API_REVIEW.md) for the complete review record and accepted limits.

## Ownership and lifecycle

- A `tensor.Tensor` owns exactly one independently releasable `storage.Storage` handle.
- Do not copy a Tensor after first use. Use `Tensor.Retain` or a view operation to obtain another handle.
- `Close` releases only that handle and is idempotent. A retained Tensor or view remains valid until its own Close.
- The zero Tensor value owns no resources. Metadata/data methods return `errs.ErrInvalidState`; `Close` is safe and idempotent.
- Constructors and deserialization return new owning allocations. Failed construction closes partially created resources.
- Storage is accessed only through bounded ReaderAt/WriterAt calls. Neither backing bytes nor a mutable Storage handle is exposed.

## Pure Go heap storage

- `storage.HeapAllocator` is a dependency-free, zero-initializing Pure Go allocator.
- Every Retain returns a distinct handle. The shared allocation is cleared and discarded after the final Release.
- Handle lifecycle and allocation bytes are synchronized. Reads are safe concurrently; the heap implementation serializes writes.
- A zero-byte allocation is valid. Negative sizes, nil/typed-nil contexts, cancelled contexts and sizes not representable as a Go slice length are rejected.
- ReadAt and WriteAt follow standard partial-I/O semantics and never return a backing-slice alias.

## Typed data and indexing

- `From[T]` / `FromWithAllocator[T]` copy fixed-width input data into owned storage.
- `Read[T]` returns a detached packed row-major copy; `Write[T]` copies input values into storage.
- The dtype must match exactly; implicit numeric conversion is not supported.
- Numeric values use little-endian representation. Bool uses canonical bytes 0/1. Complex values are stored real component then imaginary component. Float16/BFloat16 use raw bit payload types.
- `Metadata.ElementOffset` calculates checked physical element offsets from strides. `Metadata.LinearIndex` calculates row-major logical indices.
- `At[T]` and `Set[T]` provide one-element access through any supported layout. `Tensor.ByteOffset` reports the absolute byte location without exposing memory.

## Views, aliasing and layout

- `Slice` is end-exclusive and requires a positive step. `Narrow` is a unit-step range; `Select` removes the selected axis.
- `Transpose` permutes shape and strides. Omitting axes reverses axis order; explicit axes describe output-axis to input-axis mapping.
- `Reshape` creates a view only for contiguous input and preserves element count. It fails for non-contiguous input rather than silently copying.
- `BroadcastTo` right-aligns dimensions; expanded dimensions use zero strides.
- Every Tensor view retains its backing allocation independently. Mutations are visible through every alias.
- Negative strides are not supported. Empty, scalar, contiguous and positive-stride non-contiguous tensors are supported.
- An empty view has no addressable element, so its base offset may sit at or past the source's last reachable element (for example the end position of a positive-step slice). Reads and writes on an empty view are empty operations and never access storage.
- Bulk `Write` rejects overlapping or conservatively uncertain layouts with `errs.ErrOverlappingWrite` before changing any element. `Set` remains deterministic for one explicit logical index, including on broadcast aliases.
- For non-overlapping strided layouts, bulk Write scatters row-major logical values to their physical locations. A custom Storage failure during a scatter may leave earlier elements written; there is no multi-element transaction guarantee.

## Copy and serialization

- `Copy` and `Contiguous` always allocate independent contiguous storage, even if the source is already contiguous.
- `MarshalBinary` writes a version-1 `GTNS` snapshot with dtype, rank, dimensions, payload length and logical row-major payload.
- `UnmarshalBinary` validates magic, version, dtype, shape, exact payload size and canonical Bool bytes before installing new storage. The receiver must be zero or closed; errors do not mutate an active receiver.
- Serialization is versioned but is not yet a permanent pre-1.0 checkpoint compatibility guarantee.

## Deferred work

Tensor arithmetic, CPU kernels, negative-stride views, advanced indexers, mutation transactions across multiple views, autograd integration, and stable checkpoint compatibility belong to later phases or future API review. This P1 completion does not imply those capabilities.
