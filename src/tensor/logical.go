package tensor

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// packedByteSize is the size of the tensor's logical values in row-major order.
func (m Metadata) packedByteSize() (int64, error) {
	numel, err := m.Numel()
	if err != nil {
		return 0, err
	}
	width, err := m.dtype.Size()
	if err != nil {
		return 0, err
	}
	if width > 0 && numel > maxInt64/width {
		return 0, errs.ErrShapeOverflow
	}
	return numel * width, nil
}

// elementOffsetForLinear maps a row-major logical element index to a physical
// element offset using the metadata's strides.
func (m Metadata) elementOffsetForLinear(linear int64) (int64, error) {
	numel, err := m.Numel()
	if err != nil {
		return 0, err
	}
	if linear < 0 || linear >= numel {
		return 0, errs.ErrOutOfBounds
	}
	dims, strides := m.shape.Dims(), m.strides
	var offset int64
	remaining := linear
	for axis := len(dims) - 1; axis >= 0; axis-- {
		coordinate := remaining % dims[axis]
		remaining /= dims[axis]
		stride := strides[axis]
		if stride != 0 && coordinate > maxInt64/stride {
			return 0, errs.ErrShapeOverflow
		}
		term := coordinate * stride
		if offset > maxInt64-term {
			return 0, errs.ErrShapeOverflow
		}
		offset += term
	}
	return offset, nil
}

// readLogicalBytesLocked returns a detached row-major byte sequence.
// The caller must hold t.mu for reading or writing.
func (t *Tensor) readLogicalBytesLocked() ([]byte, error) {
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	if err := t.metadata.Validate(); err != nil {
		return nil, err
	}
	total, err := t.metadata.packedByteSize()
	if err != nil {
		return nil, err
	}
	if uint64(total) > uint64(^uint(0)>>1) {
		return nil, errs.ErrShapeOverflow
	}
	result := make([]byte, int(total))
	if total == 0 {
		return result, nil
	}
	if t.metadata.IsContiguous() {
		n, readErr := t.storage.ReadAt(result, t.offset)
		if readErr != nil {
			return nil, fmt.Errorf("tensor: read storage: %w", readErr)
		}
		if n != len(result) {
			return nil, fmt.Errorf("tensor: read %d of %d bytes: %w", n, len(result), errs.ErrInvalidState)
		}
		return result, nil
	}

	numel, _ := t.metadata.Numel()
	width, _ := t.metadata.dtype.Size()
	for linear := int64(0); linear < numel; linear++ {
		elementOffset, offsetErr := t.metadata.elementOffsetForLinear(linear)
		if offsetErr != nil {
			return nil, offsetErr
		}
		delta, offsetErr := elementDeltaBytes(elementOffset, width)
		if offsetErr != nil {
			return nil, offsetErr
		}
		byteOffset, offsetErr := checkedAdd(t.offset, delta)
		if offsetErr != nil {
			return nil, offsetErr
		}
		start := int(linear * width)
		end := start + int(width)
		n, readErr := t.storage.ReadAt(result[start:end], byteOffset)
		if readErr != nil {
			return nil, fmt.Errorf("tensor: read logical element %d: %w", linear, readErr)
		}
		if n != int(width) {
			return nil, fmt.Errorf("tensor: read %d of %d bytes at logical element %d: %w", n, width, linear, errs.ErrInvalidState)
		}
	}
	return result, nil
}

// writeLogicalBytesLocked scatters packed row-major bytes into the physical
// layout. The caller must hold t.mu. Overlapping layouts reject non-empty bulk writes.
func (t *Tensor) writeLogicalBytesLocked(data []byte) error {
	if t.closed || nilDependency(t.storage) {
		return errs.ErrInvalidState
	}
	if err := t.metadata.Validate(); err != nil {
		return err
	}
	total, err := t.metadata.packedByteSize()
	if err != nil {
		return err
	}
	if int64(len(data)) != total {
		return fmt.Errorf("%w: got %d bytes, want %d", errs.ErrInvalidArgument, len(data), total)
	}
	if total == 0 {
		return nil
	}
	if t.metadata.hasOverlappingElements() {
		return errs.ErrOverlappingWrite
	}
	if t.metadata.IsContiguous() {
		n, writeErr := t.storage.WriteAt(data, t.offset)
		if writeErr != nil {
			return fmt.Errorf("tensor: write storage: %w", writeErr)
		}
		if n != len(data) {
			return fmt.Errorf("tensor: wrote %d of %d bytes: %w", n, len(data), errs.ErrInvalidState)
		}
		return nil
	}

	numel, _ := t.metadata.Numel()
	width, _ := t.metadata.dtype.Size()
	for linear := int64(0); linear < numel; linear++ {
		elementOffset, offsetErr := t.metadata.elementOffsetForLinear(linear)
		if offsetErr != nil {
			return offsetErr
		}
		delta, offsetErr := elementDeltaBytes(elementOffset, width)
		if offsetErr != nil {
			return offsetErr
		}
		byteOffset, offsetErr := checkedAdd(t.offset, delta)
		if offsetErr != nil {
			return offsetErr
		}
		start := int(linear * width)
		end := start + int(width)
		n, writeErr := t.storage.WriteAt(data[start:end], byteOffset)
		if writeErr != nil {
			return fmt.Errorf("tensor: write logical element %d: %w", linear, writeErr)
		}
		if n != int(width) {
			return fmt.Errorf("tensor: wrote %d of %d bytes at logical element %d: %w", n, width, linear, errs.ErrInvalidState)
		}
	}
	return nil
}
