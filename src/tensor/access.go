package tensor

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// ByteOffset returns the absolute byte offset in the shared allocation for a
// logical element index. It does not expose the allocation or a memory alias.
func (t *Tensor) ByteOffset(indices ...int64) (int64, error) {
	if t == nil {
		return 0, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.byteOffsetLocked(indices...)
}

func (t *Tensor) byteOffsetLocked(indices ...int64) (int64, error) {
	if t.closed || nilDependency(t.storage) {
		return 0, errs.ErrInvalidState
	}
	elementOffset, err := t.metadata.ElementOffset(indices...)
	if err != nil {
		return 0, err
	}
	width, err := t.metadata.dtype.Size()
	if err != nil {
		return 0, err
	}
	delta, err := elementDeltaBytes(elementOffset, width)
	if err != nil {
		return 0, err
	}
	return checkedAdd(t.offset, delta)
}

func (t *Tensor) readElementBytes(indices ...int64) ([]byte, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	offset, err := t.byteOffsetLocked(indices...)
	if err != nil {
		return nil, err
	}
	width, err := t.metadata.dtype.Size()
	if err != nil {
		return nil, err
	}
	raw := make([]byte, int(width))
	n, err := t.storage.ReadAt(raw, offset)
	if err != nil {
		return nil, fmt.Errorf("tensor: read element: %w", err)
	}
	if n != len(raw) {
		return nil, fmt.Errorf("tensor: read %d of %d bytes: %w", n, len(raw), errs.ErrInvalidState)
	}
	return raw, nil
}

func (t *Tensor) writeElementBytes(raw []byte, indices ...int64) error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	offset, err := t.byteOffsetLocked(indices...)
	if err != nil {
		return err
	}
	width, err := t.metadata.dtype.Size()
	if err != nil {
		return err
	}
	if int64(len(raw)) != width {
		return fmt.Errorf("%w: element byte width %d, got %d", errs.ErrInvalidArgument, width, len(raw))
	}
	n, err := t.storage.WriteAt(raw, offset)
	if err != nil {
		return fmt.Errorf("tensor: write element: %w", err)
	}
	if n != len(raw) {
		return fmt.Errorf("tensor: wrote %d of %d bytes: %w", n, len(raw), errs.ErrInvalidState)
	}
	return nil
}

// At returns one element at the specified logical multidimensional index.
// It supports contiguous, strided, transposed, sliced and broadcast views.
func At[T Element](t *Tensor, indices ...int64) (T, error) {
	var zero T
	if t == nil {
		return zero, errs.ErrInvalidArgument
	}
	metadata, err := t.Metadata()
	if err != nil {
		return zero, err
	}
	requested, err := elementDType[T]()
	if err != nil {
		return zero, err
	}
	if metadata.DType() != requested {
		return zero, fmt.Errorf("%w: tensor dtype is %s, requested %s", errs.ErrUnsupportedDType, metadata.DType(), requested)
	}
	raw, err := t.readElementBytes(indices...)
	if err != nil {
		return zero, err
	}
	values, err := decodeElements[T](raw)
	if err != nil {
		return zero, err
	}
	if len(values) != 1 {
		return zero, errs.ErrInvalidState
	}
	return values[0], nil
}

// Set writes one element at the specified logical multidimensional index.
// An indexed write remains well-defined on an overlapping broadcast view:
// other logical indices that alias the same storage observe the new value.
func Set[T Element](t *Tensor, value T, indices ...int64) error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	metadata, err := t.Metadata()
	if err != nil {
		return err
	}
	requested, err := elementDType[T]()
	if err != nil {
		return err
	}
	if metadata.DType() != requested {
		return fmt.Errorf("%w: tensor dtype is %s, value dtype is %s", errs.ErrUnsupportedDType, metadata.DType(), requested)
	}
	raw, err := encodeElements([]T{value})
	if err != nil {
		return err
	}
	return t.writeElementBytes(raw, indices...)
}
