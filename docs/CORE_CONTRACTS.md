# Core Contracts

These contracts define subsystem boundaries before implementation expands. Their initial Go APIs live in the packages listed below; they do not imply that the corresponding execution engines already exist.

## Dtype and shape

- `dtype.DType` is an explicit, validated element-type identifier with a defined byte width.
- `shape.Shape` is immutable through its public API. Dimensions are non-negative; rank zero is a scalar with one element; any zero-sized dimension produces zero elements.
- Element-count and byte-span arithmetic must detect signed integer overflow before allocation or slicing.

## Tensor metadata

A tensor descriptor defines dtype, shape, element strides, layout, storage, ownership, mutability, and placement. The P0 `tensor.Metadata` value covers dtype, shape, strides, and derived contiguous/strided layout only. It owns no storage and is not yet a public owning `Tensor` type.

Strides are measured in elements. Negative strides are not accepted by the initial P0 contract. A view must keep its backing storage alive for its entire lifetime; aliasing and overlapping-write rules must be made explicit in P1 before mutable views are exposed.

## Storage ownership

`storage.Storage` is an independently releasable handle to a shared allocation:

- `Retain` returns a distinct handle that must be released independently.
- `Release` relinquishes only the calling handle. Repeated release returns `errs.ErrStorageReleased`.
- Reads and writes after release fail. Implementations follow `io.ReaderAt`/`io.WriterAt` range semantics.
- Asynchronous work and views retain storage until their final use is complete.
- Allocation size, offsets, and lengths are validated before use; range checks must not overflow.

P0 defines the interface and range validator; the concrete allocator and buffer implementations are P1 work.

## Kernel and backend

A kernel declares its operation and primary compute dtype/layout. A backend's support decision may inspect the full invocation because some operations have different input and output signatures. Every invocation argument binds validated tensor metadata to a storage handle and base offset.

The reference implementation will define correctness. SIMD and BLAS/LAPACK implementations are optional optimized paths and must be differentially tested against the reference implementation. A backend may not retain invocation slices or borrowed storage handles after its execution call returns.

## Execution and lifecycle

Long-running operations accept `context.Context`, use bounded resources, and honor cancellation at safe boundaries. Runtime components start in dependency order and stop in reverse order. A component whose `Start` fails is responsible for cleaning up its own partial startup; the manager rolls back components that had already started. Shutdown is best-effort across all components and reports joined errors.

## Memory

- Parameter storage, caller-owned inputs/outputs, temporary activations, workspace, and distributed staging have distinct lifetimes.
- Views retain backing storage.
- A buffer cannot be reused while any view or asynchronous operation references it.
- Pooling is an optimization, not an ownership model.
- `sync.Pool` is opportunistic and must not be used for deterministic resource accounting.
- Arena reset is legal only after all allocations from the arena are dead.

## Distributed identity

A distributed identity includes run ID, worker ID, rank, world size, and membership epoch. A membership snapshot must use one run/epoch/world-size tuple, have unique worker IDs, and contain every rank exactly once. IDs are opaque values supplied by the coordinator.

Collectives must define rank membership, ordering, shape compatibility, timeout, cancellation, bounded payloads, retries/idempotency, and failure behavior. Distributed transport must not leak into ordinary local tensor operations.

## Public API

Public APIs document error behavior, concurrency safety, mutation, ownership, dtype conversion, and compatibility guarantees. Errors preserve context and support `errors.Is`/`errors.As`; ordinary invalid input must not panic.

## P0 implementation map

- `errs`: stable sentinel errors.
- `dtype`, `shape`, `tensor`: metadata and checked arithmetic.
- `storage`: ownership-aware storage and allocator interfaces.
- `compute`: kernel specification, invocation, backend interface, and pre-dispatch validation.
- `runtime`: component lifecycle manager.
- `distributed`: identity and membership validation.
- `platform`: primary and compatibility target classification.

See [`P0_IMPLEMENTATION.md`](P0_IMPLEMENTATION.md) for current limits and exit gates.
