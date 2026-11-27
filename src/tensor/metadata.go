// Package tensor defines metadata contracts. Tensor allocation and operators belong to P1.
package tensor

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
)

const maxInt64 = int64(^uint64(0) >> 1)

// Layout describes the address pattern used to traverse tensor elements.
type Layout uint8

const (
	LayoutInvalid Layout = iota
	LayoutContiguous
	LayoutStrided
)

func (l Layout) Valid() bool { return l == LayoutContiguous || l == LayoutStrided }

func (l Layout) String() string {
	switch l {
	case LayoutContiguous:
		return "contiguous"
	case LayoutStrided:
		return "strided"
	default:
		return "invalid"
	}
}

// Metadata describes dtype, dimensions, and element strides without owning memory.
type Metadata struct {
	dtype   dtype.DType
	shape   shape.Shape
	strides []int64
	layout  Layout
}

// NewMetadata creates metadata with row-major contiguous strides. Empty tensors
// use zero strides because no element address is reachable.
func NewMetadata(elementType dtype.DType, dimensions shape.Shape) (Metadata, error) {
	strides, err := contiguousStrides(dimensions)
	if err != nil {
		return Metadata{}, err
	}
	return NewStridedMetadata(elementType, dimensions, strides)
}

// NewStridedMetadata creates metadata with explicit non-negative element strides.
func NewStridedMetadata(elementType dtype.DType, dimensions shape.Shape, strides []int64) (Metadata, error) {
	if !elementType.Valid() {
		return Metadata{}, fmt.Errorf("%w: %d", errs.ErrUnsupportedDType, elementType)
	}
	if len(strides) != dimensions.Rank() {
		return Metadata{}, fmt.Errorf("%w: %d strides for rank %d", errs.ErrInvalidLayout, len(strides), dimensions.Rank())
	}
	copied := append([]int64(nil), strides...)
	for axis, stride := range copied {
		if stride < 0 {
			return Metadata{}, fmt.Errorf("%w: negative stride at axis %d", errs.ErrInvalidLayout, axis)
		}
	}
	m := Metadata{dtype: elementType, shape: dimensions, strides: copied}
	if m.isContiguous() {
		m.layout = LayoutContiguous
	} else {
		m.layout = LayoutStrided
	}
	if _, err := m.RequiredBytes(); err != nil {
		return Metadata{}, err
	}
	return m, nil
}

// Validate rechecks metadata invariants at trust boundaries.
func (m Metadata) Validate() error {
	if !m.dtype.Valid() {
		return fmt.Errorf("%w: %d", errs.ErrUnsupportedDType, m.dtype)
	}
	if len(m.strides) != m.shape.Rank() {
		return fmt.Errorf("%w: stride count does not match rank", errs.ErrInvalidLayout)
	}
	for axis, stride := range m.strides {
		if stride < 0 {
			return fmt.Errorf("%w: negative stride at axis %d", errs.ErrInvalidLayout, axis)
		}
	}
	if !m.layout.Valid() || (m.layout == LayoutContiguous) != m.isContiguous() {
		return errs.ErrInvalidLayout
	}
	_, err := m.RequiredBytes()
	return err
}

// DType returns the element dtype.
func (m Metadata) DType() dtype.DType { return m.dtype }

// Shape returns immutable shape metadata.
func (m Metadata) Shape() shape.Shape { return m.shape }

// Strides returns a copy of the strides, measured in elements rather than bytes.
func (m Metadata) Strides() []int64 { return append([]int64(nil), m.strides...) }

// Layout returns the metadata's traversal layout.
func (m Metadata) Layout() Layout { return m.layout }

// Numel returns the checked number of logical tensor elements.
func (m Metadata) Numel() (int64, error) { return m.shape.Numel() }

// RequiredBytes returns the byte span from the first element through the last
// reachable element. It accounts for non-contiguous strides and excludes any
// base-storage offset, which is supplied separately by compute.Argument.
func (m Metadata) RequiredBytes() (int64, error) {
	numel, err := m.shape.Numel()
	if err != nil {
		return 0, err
	}
	if numel == 0 {
		return 0, nil
	}
	last := int64(0)
	dims := m.shape.Dims()
	for axis, dim := range dims {
		if dim <= 1 {
			continue
		}
		steps := dim - 1
		stride := m.strides[axis]
		if stride != 0 && steps > maxInt64/stride {
			return 0, errs.ErrShapeOverflow
		}
		term := steps * stride
		if last > maxInt64-term {
			return 0, errs.ErrShapeOverflow
		}
		last += term
	}
	if last == maxInt64 {
		return 0, errs.ErrShapeOverflow
	}
	elementBytes, err := m.dtype.Size()
	if err != nil {
		return 0, err
	}
	span := last + 1
	if span > maxInt64/elementBytes {
		return 0, errs.ErrShapeOverflow
	}
	return span * elementBytes, nil
}

func (m Metadata) isContiguous() bool {
	numel, err := m.shape.Numel()
	if err != nil {
		return false
	}
	if numel == 0 {
		return true
	}
	dims := m.shape.Dims()
	expected := int64(1)
	for axis := len(dims) - 1; axis >= 0; axis-- {
		if dims[axis] > 1 && m.strides[axis] != expected {
			return false
		}
		if dims[axis] > 1 && expected > maxInt64/dims[axis] {
			return false
		}
		expected *= dims[axis]
	}
	return true
}

func contiguousStrides(dimensions shape.Shape) ([]int64, error) {
	dims := dimensions.Dims()
	numel, err := dimensions.Numel()
	if err != nil {
		return nil, err
	}
	if numel == 0 {
		return make([]int64, len(dims)), nil
	}

	strides := make([]int64, len(dims))
	stride := int64(1)
	for axis := len(dims) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		if dims[axis] == 0 {
			stride = 0
			continue
		}
		if dims[axis] > 1 && stride > maxInt64/dims[axis] {
			return nil, errs.ErrShapeOverflow
		}
		stride *= dims[axis]
	}
	return strides, nil
}
