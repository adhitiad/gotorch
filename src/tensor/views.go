package tensor

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
)

// viewLocked creates a separately retained Tensor handle. The caller must hold
// t.mu for reading or writing. The view's reachable span must remain inside
// the source Tensor's bounding storage span.
func (t *Tensor) viewLocked(metadata Metadata, offset int64) (*Tensor, error) {
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	if err := metadata.Validate(); err != nil {
		return nil, err
	}
	sourceSpan, err := t.metadata.RequiredBytes()
	if err != nil {
		return nil, err
	}
	viewSpan, err := metadata.RequiredBytes()
	if err != nil {
		return nil, err
	}
	sourceEnd, err := checkedAdd(t.offset, sourceSpan)
	if err != nil {
		return nil, err
	}
	viewEnd, err := checkedAdd(offset, viewSpan)
	if err != nil {
		return nil, err
	}
	// An empty view has no addressable element. Its base offset may therefore sit
	// past the source's last reachable element (for example the end position of a
	// positive-step slice) without being an out-of-bounds access. Non-empty views
	// must stay inside the source's reachable span.
	if offset < t.offset {
		return nil, errs.ErrOutOfBounds
	}
	if viewSpan > 0 && (offset > sourceEnd || viewEnd > sourceEnd) {
		return nil, errs.ErrOutOfBounds
	}
	handle, err := t.storage.Retain()
	if err != nil {
		return nil, fmt.Errorf("tensor: retain view storage: %w", err)
	}
	if nilDependency(handle) || handle.Size() != t.storage.Size() {
		if !nilDependency(handle) {
			_ = handle.Release()
		}
		return nil, fmt.Errorf("%w: storage returned an invalid view handle", errs.ErrInvalidState)
	}
	return &Tensor{metadata: metadata, storage: handle, offset: offset}, nil
}

func checkedAdd(a, b int64) (int64, error) {
	if a < 0 || b < 0 || a > maxInt64-b {
		return 0, errs.ErrShapeOverflow
	}
	return a + b, nil
}

func elementDeltaBytes(elementOffset, width int64) (int64, error) {
	if elementOffset < 0 || width <= 0 || elementOffset > maxInt64/width {
		return 0, errs.ErrShapeOverflow
	}
	return elementOffset * width, nil
}

func (t *Tensor) viewOffsetLocked(elementDelta int64) (int64, error) {
	width, err := t.metadata.dtype.Size()
	if err != nil {
		return 0, err
	}
	delta, err := elementDeltaBytes(elementDelta, width)
	if err != nil {
		return 0, err
	}
	return checkedAdd(t.offset, delta)
}

// Slice creates an end-exclusive positive-step view on axis. The returned
// Tensor owns an independent storage handle but aliases the source allocation.
func (t *Tensor) Slice(axis int, start, end, step int64) (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	if axis < 0 || axis >= t.metadata.shape.Rank() {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := t.metadata.Slice(axis, start, end, step)
	if err != nil {
		return nil, err
	}
	stride := t.metadata.strides[axis]
	if stride != 0 && start > maxInt64/stride {
		return nil, errs.ErrShapeOverflow
	}
	offset, err := t.viewOffsetLocked(start * stride)
	if err != nil {
		return nil, err
	}
	return t.viewLocked(metadata, offset)
}

// Narrow creates a unit-step view of length elements on axis.
func (t *Tensor) Narrow(axis int, start, length int64) (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	if axis < 0 || axis >= t.metadata.shape.Rank() {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := t.metadata.Narrow(axis, start, length)
	if err != nil {
		return nil, err
	}
	stride := t.metadata.strides[axis]
	if stride != 0 && start > maxInt64/stride {
		return nil, errs.ErrShapeOverflow
	}
	offset, err := t.viewOffsetLocked(start * stride)
	if err != nil {
		return nil, err
	}
	return t.viewLocked(metadata, offset)
}

// Select creates a view with axis removed after selecting one index.
func (t *Tensor) Select(axis int, index int64) (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	if axis < 0 || axis >= t.metadata.shape.Rank() {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := t.metadata.Select(axis, index)
	if err != nil {
		return nil, err
	}
	stride := t.metadata.strides[axis]
	if stride != 0 && index > maxInt64/stride {
		return nil, errs.ErrShapeOverflow
	}
	offset, err := t.viewOffsetLocked(index * stride)
	if err != nil {
		return nil, err
	}
	return t.viewLocked(metadata, offset)
}

// Reshape creates a view with a new shape. It requires contiguous input and
// preserves the allocation; use Copy or Contiguous before reshaping strided data.
func (t *Tensor) Reshape(dimensions ...int64) (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	metadata, err := t.metadata.Reshape(dimensions...)
	if err != nil {
		return nil, err
	}
	return t.viewLocked(metadata, t.offset)
}

// Transpose creates a permuted-stride view. With no axes, dimensions are reversed.
func (t *Tensor) Transpose(axes ...int) (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	metadata, err := t.metadata.Transpose(axes...)
	if err != nil {
		return nil, err
	}
	return t.viewLocked(metadata, t.offset)
}

// BroadcastTo creates a zero-stride aliasing view. Bulk Write is rejected when
// logical elements overlap; Set still addresses one explicit logical index.
func (t *Tensor) BroadcastTo(dimensions ...int64) (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	target, err := shape.New(dimensions...)
	if err != nil {
		return nil, err
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	metadata, err := t.metadata.BroadcastTo(target)
	if err != nil {
		return nil, err
	}
	return t.viewLocked(metadata, t.offset)
}
