# Tensor API (P1)

This document describes the supported Tensor value, indexing, view, copy and binary snapshot APIs. It does not claim tensor arithmetic or compute kernels are implemented.

## Construct and own a tensor

```go
matrix, err := tensor.From([]float32{
	1, 2, 3,
	4, 5, 6,
}, 2, 3)
if err != nil {
	return err
}
defer matrix.Close()
```

`NewTensor(dtype, dimensions...)` allocates zero-initialized contiguous storage. `NewTensorWithAllocator` accepts a caller-provided allocator. `From[T]` and `FromWithAllocator[T]` allocate and copy fixed-width typed values. With no dimensions, one input element creates a rank-zero scalar. A zero-sized dimension creates an empty tensor.

A Tensor owns one independently releasable storage handle. Do not copy a Tensor after first use; use `Retain` to obtain another Tensor handle. Every view and retained Tensor has its own close lifecycle. `Close` is idempotent. The zero value owns no allocation; data and metadata operations return `errs.ErrInvalidState`, while `Close` is safe.

## Typed access and indexing

```go
value, err := tensor.At[float32](matrix, 1, 2) // 6
if err != nil {
	return err
}
if err := tensor.Set(matrix, float32(42), 1, 2); err != nil {
	return err
}
values, err := tensor.Read[float32](matrix) // detached row-major copy
if err != nil {
	return err
}
```

The requested Go type must match the Tensor dtype exactly; no implicit numeric conversion occurs. `Read` returns a detached copy and `Write` copies the supplied values. Both operate in logical row-major order, including for supported strided views.

`Metadata.ElementOffset(indices...)` returns the checked physical offset in elements, applying the metadata strides. `Metadata.LinearIndex(indices...)` returns the logical row-major linear index regardless of physical layout. `Tensor.ByteOffset(indices...)` returns an absolute byte offset inside the shared allocation without exposing its bytes.

All index ranks and coordinates are checked. Invalid indices return an error compatible with `errs.ErrOutOfBounds`; arithmetic overflow returns `errs.ErrShapeOverflow`.

## Views and aliasing

```go
transposed, err := matrix.Transpose()
if err != nil {
	return err
}
defer transposed.Close()

row, err := matrix.Narrow(0, 0, 1)
if err != nil {
	return err
}
defer row.Close()

column, err := matrix.Select(1, 1)
if err != nil {
	return err
}
defer column.Close()
```

- `Slice(axis, start, end, step)` uses an end-exclusive range and requires `step > 0`.
- `Narrow(axis, start, length)` is a unit-step slice.
- `Select(axis, index)` removes the selected axis.
- `Transpose(axes...)` permutes axes; no arguments reverse all axes. Explicit axes map each output axis to its source axis.
- `Reshape(dimensions...)` changes only metadata and therefore requires contiguous input with the same element count.
- `BroadcastTo(dimensions...)` aligns dimensions from the right and represents expanded axes with zero strides.

Every view retains the backing allocation. Closing the source does not invalidate an existing view; closing a view does not close the source. Writes through one alias are visible through other aliases.

Negative strides are not supported. Slicing never reverses an axis. Reshaping a non-contiguous view returns `errs.ErrUnsupportedOperation`; call `Copy` or `Contiguous` first.

### Overlapping writes

Broadcast views may map multiple logical indices to one physical element. `Read`, `At` and `Set` are supported. Bulk `Write` is rejected before modifying storage when the layout is detected as overlapping or cannot be proven non-overlapping; the error is `errs.ErrOverlappingWrite`. `Set` targets one explicit logical index, so any aliases of that physical element observe the update.

For non-overlapping strided views such as ordinary transposes and slices, `Write` scatters logical values through the physical strides. Multi-element writes to strided storage are not transactional if a custom Storage implementation fails partway through; callers must coordinate concurrent mutation.

## Copy and contiguous conversion

```go
dense, err := transposed.Contiguous()
if err != nil {
	return err
}
defer dense.Close()

independent, err := matrix.Copy()
if err != nil {
	return err
}
defer independent.Close()
```

Both operations produce independent row-major storage. They copy even if the source is already contiguous. Mutating the returned tensor never changes the source.

## Binary snapshots

`Tensor.MarshalBinary` implements `encoding.BinaryMarshaler`. The version-1 `GTNS` format contains magic, version, dtype, rank, signed 64-bit dimensions, payload length and canonical row-major logical bytes. It does not serialize an allocator, storage handle, view offset or stride arrangement.

`(*Tensor).UnmarshalBinary` validates the full input before replacing the receiver. The receiver must be a zero or closed Tensor. On error, an existing receiver remains unchanged. Decoding always creates independent contiguous storage. Boolean payloads must use canonical bytes 0 and 1.

The format is versioned but remains a pre-1.0 GoTorch format; applications should not treat it as a permanent cross-version checkpoint guarantee yet.

## Concurrency and lifetime

Tensor methods synchronize access to each handle's lifecycle. The Pure Go heap storage synchronizes byte access and protects shared allocation lifetime. Concurrent readers are safe. Callers must coordinate conflicting mutations across aliases; the API does not promise an atomic snapshot across a multi-element strided read or write. Views retain storage until their own final `Close`.
